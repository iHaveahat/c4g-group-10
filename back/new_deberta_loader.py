import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "glurgle/deberta_satire"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABEL_MAP = {0: "🟩 Real News", 1: "🟥 Fake News", 2: "🟨 Satire"}

@torch.inference_mode()
def predict(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1)
    pred   = int(probs.argmax())
    return {"result": LABEL_MAP[pred], "confidence": round(float(probs[0, pred]) * 100, 2)}

# quick test
if __name__ == "__main__":
    print(predict(input("Text to check: ")))
