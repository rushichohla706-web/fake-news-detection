"""TruthCheck AI application backend."""

import json
import joblib
import os
import re
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlparse

from flask import Flask, g, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")


def resolve_model_path(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, "model", filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_PATH = resolve_model_path("model.pkl")
VECTORIZER_PATH = resolve_model_path("vectorizer.pkl")
MODEL_INFO_PATH = resolve_model_path("model_info.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    MODEL = joblib.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    VECTORIZER = joblib.load(f)

MODEL_INFO = {"model_name": "logistic_regression", "accuracy": 0.9}
if os.path.exists(MODEL_INFO_PATH):
    with open(MODEL_INFO_PATH, "rb") as f:
        MODEL_INFO = joblib.load(f)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source TEXT,
            result TEXT,
            confidence REAL,
            explanation TEXT,
            evidence TEXT,
            contradicting_evidence TEXT,
            source_credibility TEXT,
            publication_date TEXT,
            language TEXT,
            ai_warning TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def clean_text(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def detect_language(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["the ", "and ", "that ", "with ", "for "]):
        return "English"
    if any(token in lowered for token in ["el ", "la ", "y ", "que ", "por "]):
        return "Spanish"
    if any(token in lowered for token in ["le ", "la ", "et ", "une ", "des "]):
        return "French"
    if any(token in lowered for token in ["der ", "und ", "die ", "ist "]):
        return "German"
    return "English"


def extract_publication_date(text: str) -> str:
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def detect_ai_warning(text: str) -> str:
    suspicious = [
        "as an ai language model",
        "in this article",
        "let me explain",
        "as a large language model",
        "this content was generated",
        "the following is generated",
    ]
    lowered = text.lower()
    if any(pattern in lowered for pattern in suspicious):
        return "AI-generated wording patterns detected. Independent corroboration is recommended."
    return "No strong AI-writing pattern detected. Independent verification is still essential."


def source_credibility(text: str):
    lowered = text.lower()
    trusted = ["reuters", "apnews", "bbc", "who", "nasa", "nytimes", "theguardian", "aljazeera", "cbc", "wsj"]
    suspicious = ["shocking", "must see", "viral", "secret", "miracle", "click here", "banned", "you won't believe"]
    score = 0.5
    for domain in trusted:
        if domain in lowered:
            score += 0.3
    for term in suspicious:
        if term in lowered:
            score -= 0.25
    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        return "High", score
    if score >= 0.4:
        return "Medium", score
    return "Low", score


def evidence_signals(text: str):
    lowered = text.lower()
    reliable_markers = [
        "according to",
        "official statement",
        "reported by",
        "study shows",
        "data from",
        "government said",
        "confirmed by",
        "analysis shows",
    ]
    suspicious_markers = [
        "shocking",
        "secret",
        "miracle",
        "viral",
        "must see",
        "you won't believe",
        "click here",
        "exposed",
    ]
    conflict_markers = ["however", "but", "contradict", "conflict", "disputed", "unclear", "unverified"]
    reliable_score = sum(1 for marker in reliable_markers if marker in lowered) / len(reliable_markers)
    suspicious_score = sum(1 for marker in suspicious_markers if marker in lowered) / len(suspicious_markers)
    conflicting = any(marker in lowered for marker in conflict_markers)
    return {
        "reliable_source_score": reliable_score,
        "sensational_score": suspicious_score,
        "conflicting_evidence": conflicting,
        "reliable_markers": [marker for marker in reliable_markers if marker in lowered][:3],
        "suspicious_markers": [marker for marker in suspicious_markers if marker in lowered][:3],
    }


def extract_claims(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if 25 <= len(cleaned) <= 220 and cleaned:
            claims.append(cleaned)
    return claims[:4]


def normalize_prediction(raw_prediction: str, confidence: float, signals: dict) -> str:
    if signals.get("conflicting_evidence"):
        return "Unverified"
    if raw_prediction == "REAL":
        if confidence >= 0.68 and signals.get("reliable_source_score", 0) >= 0.28:
            return "Credible"
        return "Misleading"
    if confidence >= 0.68 and signals.get("sensational_score", 0) >= 0.18:
        return "False"
    if confidence >= 0.55:
        return "Misleading"
    return "Unverified"


def classify_text(text: str):
    article_text = clean_text(text)
    if not article_text or len(article_text) < 10:
        raise ValueError("Please provide at least 10 characters of article text.")

    vectorized = VECTORIZER.transform([article_text])
    classes = list(MODEL.classes_)
    probabilities = MODEL.predict_proba(vectorized)[0]
    real_probability = float(probabilities[classes.index("REAL")])
    raw_prediction = "REAL" if real_probability > 0.50 else "FAKE"
    confidence = real_probability if raw_prediction == "REAL" else 1 - real_probability

    confidence = max(0.15, min(0.96, confidence))
    signals = evidence_signals(article_text)
    source_rating, source_value = source_credibility(article_text)
    adjusted_confidence = max(0.15, min(0.96, confidence + (source_value * 0.2) + (signals["reliable_source_score"] * 0.25) - (signals["sensational_score"] * 0.25)))
    verdict = normalize_prediction(raw_prediction, adjusted_confidence, signals)

    if verdict == "Credible":
        explanation = "The content contains patterns consistent with a sourced, fact-based report, but the model should still be treated as advisory."
        recommendation = "Treat as credible but verify with official records or primary reporting before acting on it."
    elif verdict == "False":
        explanation = "The language patterns and weak evidence signals are consistent with a false or highly misleading claim."
        recommendation = "Exercise caution and confirm the claim with multiple independent, trustworthy sources."
    elif verdict == "Misleading":
        explanation = "The article may contain a partly framed or selectively reported claim that lacks strong corroboration."
        recommendation = "Cross-check with a primary source or an independent outlet before sharing it further."
    else:
        explanation = "The claim could not be confirmed using enough reliable evidence, so the status remains uncertain."
        recommendation = "Treat this claim cautiously and seek additional reliable evidence before accepting it as fact."

    supporting = []
    if signals["reliable_markers"]:
        supporting.extend([f"Corroboration signal detected: {marker}." for marker in signals["reliable_markers"]])
    else:
        supporting.append("No strong corroboration markers were identified in the supplied text.")

    contradicting = []
    if signals["conflicting_evidence"]:
        contradicting.append("Conflicting or qualifying language is present, which lowers certainty.")
    if signals["suspicious_markers"]:
        contradicting.extend([f"Suspicious phrasing detected: {marker}." for marker in signals["suspicious_markers"]])
    if not contradicting:
        contradicting.append("No direct contradiction pattern was detected in the supplied text.")

    result = {
        "prediction": verdict,
        "confidence": round(adjusted_confidence * 100, 1),
        "reason": explanation,
        "recommendation": recommendation,
        "important_claims": extract_claims(article_text),
        "evidence_supporting": supporting,
        "evidence_contradicting": contradicting,
        "source_credibility": source_rating,
        "publication_date": extract_publication_date(article_text),
        "detected_language": detect_language(article_text),
        "ai_generated_warning": detect_ai_warning(article_text),
        "source_url": "",
        "title": article_text[:110] or "Untitled Article",
        "model_prediction": raw_prediction,
        "model_confidence": round(confidence * 100, 1),
    }
    return result


def persist_analysis(result: dict, article_text: str, source_url: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO analysis_history (
            title, source, result, confidence, explanation, evidence, contradicting_evidence,
            source_credibility, publication_date, language, ai_warning, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("title") or article_text[:110],
            source_url or "User Input",
            result["prediction"],
            result["confidence"],
            result["reason"],
            json.dumps(result["evidence_supporting"]),
            json.dumps(result["evidence_contradicting"]),
            result["source_credibility"],
            result["publication_date"],
            result["detected_language"],
            result["ai_generated_warning"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_INFO.get("model_name"))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    article_text = clean_text(payload.get("text") or "")
    source_url = (payload.get("url") or "").strip()

    if not article_text and source_url:
        parsed = urlparse(source_url)
        if not parsed.scheme or not parsed.netloc:
            return jsonify({"error": "Please enter a valid URL or paste article text."}), 400
        article_text = source_url

    if not article_text or len(article_text) < 10:
        return jsonify({"error": "Please paste at least 10 characters of article text or provide a valid news URL."}), 400

    try:
        result = classify_text(article_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result["source_url"] = source_url
    if source_url:
        parsed = urlparse(source_url)
        result["title"] = result["title"] if result["title"] else (parsed.netloc or "News Link")

    persist_analysis(result, article_text, source_url)

    db = get_db()
    db.execute(
        "INSERT INTO predictions (text, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
        (article_text[:500], result["prediction"], result["confidence"] / 100.0, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()

    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    return analyze()


@app.route("/predict", methods=["POST"])
def simple_predict():
    payload = request.get_json(silent=True) or {}
    article_text = payload.get("text")

    if not isinstance(article_text, str) or len(article_text.strip()) < 10:
        return jsonify({"error": "Please provide at least 10 characters of news text."}), 400

    cleaned_text = clean_text(article_text)
    vectorized = VECTORIZER.transform([cleaned_text])
    probabilities = MODEL.predict_proba(vectorized)[0]
    classes = list(MODEL.classes_)
    real_probability = float(probabilities[classes.index("REAL")])
    probability_score = real_probability if real_probability > 0.50 else 1 - real_probability
    signals = evidence_signals(cleaned_text)

    if signals["conflicting_evidence"] or probability_score < 0.60:
        verdict = "MISLEADING"
        reason = "The claim has conflicting signals or insufficient model certainty to confirm it as fully real or fake."
        recommendation = "Verify the claim with official records and multiple independent, trustworthy sources."
    elif real_probability > 0.50:
        verdict = "REAL"
        reason = "The text matches patterns associated with real news and its predicted REAL probability is above 0.50."
        recommendation = "Confirm the report with an official source before relying on or sharing it."
    else:
        verdict = "FAKE"
        reason = "The text matches patterns associated with fake news and its predicted REAL probability is 0.50 or lower."
        recommendation = "Do not share it until the claim is confirmed by reliable independent sources."

    return jsonify({
        "verdict": verdict,
        "confidence_score": round(probability_score * 100, 1),
        "reason": reason,
        "recommendation": recommendation,
    })


@app.route("/api/history")
def history():
    db = get_db()
    rows = db.execute(
        "SELECT id, title, source, result, confidence, explanation, publication_date, source_credibility, created_at FROM analysis_history ORDER BY id DESC LIMIT 12"
    ).fetchall()
    return jsonify([
        {
            "id": row["id"],
            "title": row["title"] or "Untitled",
            "source": row["source"] or "User Input",
            "result": row["result"],
            "confidence": row["confidence"],
            "explanation": row["explanation"],
            "publication_date": row["publication_date"],
            "source_credibility": row["source_credibility"],
            "created_at": row["created_at"],
        }
        for row in rows
    ])


@app.route("/api/history/<int:entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    db = get_db()
    db.execute("DELETE FROM analysis_history WHERE id = ?", (entry_id,))
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/stats")
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS c FROM analysis_history").fetchone()["c"]
    credible = db.execute("SELECT COUNT(*) AS c FROM analysis_history WHERE result = 'Credible'").fetchone()["c"]
    misleading = db.execute("SELECT COUNT(*) AS c FROM analysis_history WHERE result = 'Misleading'").fetchone()["c"]
    false = db.execute("SELECT COUNT(*) AS c FROM analysis_history WHERE result = 'False'").fetchone()["c"]
    unverified = db.execute("SELECT COUNT(*) AS c FROM analysis_history WHERE result = 'Unverified'").fetchone()["c"]
    avg_conf = db.execute("SELECT AVG(confidence) AS a FROM analysis_history").fetchone()["a"] or 0
    return jsonify(
        {
            "total": total,
            "credible": credible,
            "misleading": misleading,
            "false": false,
            "unverified": unverified,
            "avg_confidence": round(avg_conf, 1),
            "model_name": MODEL_INFO.get("model_name"),
            "model_accuracy": round((MODEL_INFO.get("accuracy") or 0) * 100, 1),
        }
    )

def predict_news(news_text):
    transformed_text = VECTORIZER.transform([news_text])
    return MODEL.predict(transformed_text)[0]

if __name__ == "__main__":
    init_db()
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
