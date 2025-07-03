import torch
from transformers import AutoTokenizer
from model_def import NeuralNetwork
import pandas as pd

# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def predict_text(text: str) -> dict:
    if pd.isna(text) or text.strip() == "":
        return {"result": "❌ Empty text", "confidence": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=256
    )

    with torch.no_grad():
        logits = model(inputs["input_ids"])
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label = "🟩 Real News" if pred == 0 else "🟥 Fake News"
    return {"result": label, "confidence": round(conf * 100, 2)}
