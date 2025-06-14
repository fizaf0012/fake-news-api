from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
print("Working Directory:", os.getcwd())

# ---------------------------
# Load Model and TF-IDF Vectorizer
# ---------------------------
MODEL_PATH = r"C:\Users\Hassan\Desktop\ffb\012\best_fake_news_model.pkl"
VECTORIZER_PATH = r"C:\Users\Hassan\Desktop\ffb\012\tfidf_vectorizer.pkl"


if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("✅ ERROR: Model or Vectorizer file not found. Ensure they exist in the same directory.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    tfidf = pickle.load(file)


print("✅ Model and TF-IDF Vectorizer loaded successfully.")

# ---------------------------
# Initialize FastAPI App
# ---------------------------
app = FastAPI(title="Fake News Detection API")

# ---------------------------
# Define Request Model for /predict
# ---------------------------
class NewsText(BaseModel):
    text: str

# ---------------------------
# Define Endpoints
# ---------------------------
@app.get("/media")
def media_check():
    return {"status": "✅ API is running successfully."}

@app.post("/predict")
def predict(data: NewsText):
    transformed = tfidf.transform([data.text])
    prediction = model.predict(transformed)
    label = "Real News" if prediction[0] == 1 else "Fake News"
    return {"prediction": label}

