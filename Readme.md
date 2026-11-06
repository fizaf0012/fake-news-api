# Fake News Detection

A machine learning system that classifies news text as **Real** or **Fake**, served through a FastAPI prediction endpoint and packaged with Docker and a GitHub Actions workflow.

Originally developed as a Big Data Analytics project: text processing and model training with PySpark, with the final model exposed as an API.

## Overview

| | |
|---|---|
| **Task** | Binary text classification (Real vs Fake news) |
| **Features** | TF-IDF vectorization |
| **Models compared** | Logistic Regression, Random Forest |
| **Serving** | FastAPI + Uvicorn |
| **Packaging** | Docker, GitHub Actions |

## Dataset

- **Source:** `<dataset name and link, e.g. Kaggle "Fake and Real News Dataset">`
- **Size:** `<number of articles>` (`<real count>` real, `<fake count>` fake)
- **Split:** `<e.g. 80/20 train/test>`

## Approach

1. **Preprocessing:** text cleaning and tokenization (`<describe: lowercasing, stop-word removal, etc.>`).
2. **Feature extraction:** TF-IDF turns each article into a sparse weighted term vector.
3. **Training:** Logistic Regression and Random Forest were trained and compared on the same split.
4. **Selection:** the best-performing model was saved as `best_fake_news_model.pkl` together with the fitted `tfidf_vectorizer.pkl`.
5. **Serving:** the API loads both files and classifies incoming text.

## Results

Evaluated on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | `<xx.x%>` | `<xx.x%>` | `<xx.x%>` | `<xx.x%>` |
| Random Forest | `<xx.x%>` | `<xx.x%>` | `<xx.x%>` | `<xx.x%>` |

![Results](screenshots/results.png)

> **Note on generalization:** public fake news datasets often score very high because articles from the two classes differ in source-specific patterns (datelines, outlet names, formatting). Scores on this dataset may not carry over to news from other sources.

## API

### `POST /predict`

Request:

```json
{ "text": "Paste the news article text here" }
```

Response:

```json
{ "prediction": "Fake News", "confidence": 0.93 }
```

`confidence` is included when the saved model supports probability estimates.

### `GET /health`

Returns `{"status": "API is running"}`.

Interactive docs are available at `http://localhost:8000/docs` once the server is running.

## Run locally

```bash
git clone https://github.com/fizaf0012/fake-news-api.git
cd fake-news-api
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://localhost:8000/docs.

### With Docker

```bash
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api
```

## Project structure

```
fake-news-api/
├── main.py                      # FastAPI app
├── best_fake_news_model.pkl     # trained classifier
├── tfidf_vectorizer.pkl         # fitted TF-IDF vectorizer
├── requirements.txt
├── Dockerfile
├── .github/workflows/           # CI workflow
└── screenshots/                 # results and API screenshots
```

## Tech stack

Python · PySpark · scikit-learn · TF-IDF · FastAPI · Uvicorn · Docker · GitHub Actions · AWS S3

## Author

**Fiza Fatima** · [GitHub](https://github.com/fizaf0012) · [LinkedIn](https://www.linkedin.com/in/fiza-fatima-aa2b9a272/)
