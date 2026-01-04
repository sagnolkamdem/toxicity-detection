import torch
import time
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(
    title="RoBERTa Toxicity API",
    description="A local API to detect toxic comments using a finetuned RoBERTa model.",
    version="1.0"
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Calculate duration
    process_time = time.time() - start_time

    # Add a custom header to the response
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    return response

MODEL_PATH = "../models"

print(f"Loading model from {MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise e


class ToxicityRequest(BaseModel):
    text: str


# ENDPOINTS
@app.get("/")
def home():
    return {"status": "online", "message": "RoBERTa Toxicity API is running!"}


@app.post("/predict")
def predict(request: ToxicityRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Tokenize
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)

    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 3. Calculate Probabilities
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    prob_toxic = float(probs[1])

    # 4. Determine Label
    is_toxic = prob_toxic >= 0.5

    return {
        "text": request.text,
        "prediction": "Toxic" if is_toxic else "Non-Toxic",
        "confidence": prob_toxic,
        "is_toxic": is_toxic
    }


if __name__ == "__main__":
    import uvicorn
    # localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)