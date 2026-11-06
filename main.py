"""Fake News Detection API.

Serves a trained TF-IDF + classifier pipeline over HTTP.
Run locally:  uvicorn main:app --reload
"""
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent


def find_file(*names: str) -> Path:
    """Return the first candidate file that exists next to this script."""
    for name in names:
        path = BASE_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {names} found in {BASE_DIR}. "
        "Place the trained model and vectorizer next to main.py."
    )


# Accepts both the correct name and the double-extension name currently in the repo.
MODEL_PATH = find_file("best_fake_news_model.pkl", "best_fake_news_model.pkl.pkl")
VECTORIZER_PATH = find_file("tfidf_vectorizer.pkl", "tfidf_vectorizer.pkl.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)

app = FastAPI(title="Fake News Detection API")


class NewsText(BaseModel):
    text: str = Field(..., min_length=1, description="News article text to classify")


@app.get("/health")
def health():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: NewsText):
    if not data.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    features = tfidf.transform([data.text])
    prediction = model.predict(features)[0]
    result = {"prediction": "Real News" if prediction == 1 else "Fake News"}

    # Add a confidence score when the model supports probabilities.
    if hasattr(model, "predict_proba"):
        result["confidence"] = round(float(model.predict_proba(features)[0].max()), 4)

    return result
