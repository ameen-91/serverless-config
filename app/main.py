from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from typing import List
from typing_extensions import Annotated
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_detection = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# Initialize FastAPI app
app = FastAPI()


# Define a request model to ensure valid input with Annotated
class TextRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(min_length=1, strip_whitespace=True),
    ]


# Define a batch request model to handle multiple texts at once with Annotated


@app.post("/predict/")
async def predict_sentiment(request: TextRequest):
    try:
        # Perform sentiment detection
        result = sentiment_detection(request.text)
        return {"result": result}
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
