from back.deberta_loader import predict_deberta, model
from predictor import predict_text

modelUse = "tfif"
def predict_fake_news_with_confidence(text: str) -> dict:
    """
    Routes prediction to the model.
    """
    if modelUse == "tfidf":
        return predict_text(text)
    # return {"result": "❌ Empty text", "confidence": 0.0}
    else:
         return predict_deberta(text)
