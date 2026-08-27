import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "news_dataset.csv"),
    os.path.join(BASE_DIR, "news_dataset.csv"),
]
DATA_PATH = next((path for path in DATA_CANDIDATES if os.path.exists(path)), DATA_CANDIDATES[0])
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=5000, ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Random forest provides class probabilities for the serving threshold.
models = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
}

best_model = None
best_name = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"  {name}: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# Save artifacts
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    joblib.dump(best_model, f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    joblib.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "model_info.pkl"), "wb") as f:
    joblib.dump({"model_name": best_name, "accuracy": best_acc}, f)

print(f"\n✅ Best model: {best_name} (accuracy: {best_acc:.4f})")
print("   Saved to model/ folder")