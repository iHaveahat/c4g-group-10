import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
model_path = os.path.join(os.path.dirname(__file__), "deberta_fake_news_model")
hub_model_id = "glurgle/deberta_fake_news_model"
tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
model = AutoModelForSequenceClassification.from_pretrained(hub_model_id).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_deberta(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label = "🟩 Real News" if pred == 0 else "🟥 Fake News"
    return {
        "result": label,
        "confidence": round(conf * 100, 2)
    }
