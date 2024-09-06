from fastapi import FastAPI

from transformers import pipeline

sentiment_detection = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

app = FastAPI()


@app.post("/predict_sentiment/")
def predict_sentiment(text: str):
    result = sentiment_detection(text)
    return {"result": result}
