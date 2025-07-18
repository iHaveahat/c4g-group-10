from back.deberta_loader import predict_deberta, model
from predictor import predict_text
from back.new_deberta_loader import predict, model
modelUse = "tff"
def predict_fake_news_with_confidence(text: str) -> dict:
    """
    Routes prediction to the model.
    """
    if modelUse == "tfidf":
        return predict_text(text)
    # return {"result": "❌ Empty text", "confidence": 0.0}
    elif modelUse=="tfif":
         return predict_deberta(text)
    else:
        return predict(text)
