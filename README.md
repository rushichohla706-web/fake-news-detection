# Veritas Desk — AI Fake News Detector

A full-stack fake news detection app: Flask backend, scikit-learn ML pipeline
(TF-IDF + Random Forest), SQLite for history/analytics,
and a custom HTML/CSS/JS dashboard.

## Stack
- **Backend:** Python, Flask
- **ML:** scikit-learn (TF-IDF vectorizer and Random Forest classifier with
  probability-based REAL/FAKE classification),
  pandas / numpy for data handling
- **Database:** SQLite (swap for MySQL by changing the `sqlite3` calls in
  `app.py` to a MySQL driver such as `PyMySQL` or `mysql-connector-python` —
  the schema is a single flat table so the migration is a small change)
- **Frontend:** vanilla HTML/CSS/JS (no framework), talking to the backend
  through a small JSON API

## Project structure
```
fake_news_detector/
├── app.py                  # Flask app, API routes, SQLite access
├── train_model.py          # Trains & evaluates the ML models, saves artifacts
├── requirements.txt
├── data/
│   ├── generate_dataset.py # Builds the labeled training CSV
│   └── news_dataset.csv    # Generated training data (1,040 rows)
├── model/
│   ├── model.pkl           # Trained Random Forest classifier
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── model_info.pkl      # Metadata: which model won, accuracy scores
├── database/
│   └── app.db               # SQLite database (created on first run)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js
```

## Setup

```bash
cd fake_news_detector
pip install -r requirements.txt

# (Optional) regenerate the dataset and retrain the models
python generate_dataset.py
python train_model.py

# Run the app
python app.py
```

Visit **http://localhost:5000**.

## How it works
1. **Training** (`train_model.py`): loads `data/news_dataset.csv`, splits
  80/20 train/test, fits a TF-IDF vectorizer (unigrams + bigrams, English
  stop words removed, top 5,000 features), and trains a Random Forest
  classifier. The model is evaluated on the held-out test set and saved as
  the production model.
2. **Serving** (`app.py`): loads the pickled vectorizer + model once at
   startup. `POST /api/predict` cleans the submitted text, vectorizes it,
   predicts REAL/FAKE with a confidence score, and logs the result to
   SQLite.
3. **Dashboard**: the UI shows the verdict as an animated "stamp,"
   a live confidence bar, aggregate stats (total checked, verified,
   flagged, average confidence, model accuracy), and a ledger of the most
   recent checks with delete support.

## About the training data
Because this environment has no network access, `data/generate_dataset.py`
synthesizes a labeled corpus from real-news templates (attributed,
hedged, source-quoting wire-service style) and fake-news templates
(sensational, clickbait, conspiracy-style phrasing) across 26 topics —
1,040 rows total, balanced 50/50. This is enough to demonstrate a working,
accurate pipeline end-to-end. **Before using this in production, retrain
on a real, larger labeled dataset** (e.g. LIAR, FakeNewsNet, or a
Kaggle fake/real news corpus) so the model generalizes beyond template
phrasing — swap the CSV in `data/` and rerun `train_model.py`, no other
code changes needed.

## Notes for production / company use
- Swap SQLite for MySQL/Postgres for concurrent multi-user write load.
- Add authentication (e.g. Flask-Login) before exposing the history/ledger
  endpoints outside a trusted network.
- Put the app behind a WSGI server (gunicorn/uwsgi) instead of the Flask
  dev server, and disable `debug=True`.
- Consider periodic retraining and a model registry if you'll be updating
  the classifier over time.
