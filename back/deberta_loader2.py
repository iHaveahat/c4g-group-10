import torch
from transformers import AutoTokenizer
import os

model_dir = os.path.join(os.path.dirname(__file__), "deberta_quantized_model_int8")
model_path = os.path.join(model_dir, "best_model.pt")

# Load tokenizer normally
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load TorchScript quantized model
model = torch.jit.load(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_deberta(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # TorchScript models typically expect dict inputs for forward
        outputs = model(**inputs)
        
        # outputs may be a tensor or tuple, depending on the model, adjust accordingly
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label = "🟩 Real News" if pred == 0 else "🟥 Fake News"
    return {
        "result": label,
        "confidence": round(conf * 100, 2)
    }
