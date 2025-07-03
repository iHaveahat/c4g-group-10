# fake_news_api.py
# This Flask application serves as the API backend for the fake news detector.


from flask import Flask, request, jsonify, render_template
import sys
import os


try:
    import backend # This imports your team's TF-IDF backend
    print("Team's backend.py (TF-IDF model) imported successfully!")
except ImportError:
    print("ERROR: Could not import team's backend.py. Make sure it's in the same directory or adjust path.")
    print("Please ensure backend.py and its associated model.pkl are present.")
    sys.exit(1) # Exit if the core backend isn't found, as this API depends on it.


# You will need to install these: pip install newspaper3k trafilatura
from newspaper import Article
import trafilatura
import re

# --- CORS for Frontend Communication ---
# This is crucial when your frontend (Next.js) is running on a different port/origin
# (e.g., Next.js on localhost:3000, Flask on localhost:5000).
# pip install flask-cors
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes (you can configure it more specifically if needed)


import requests

def get_text_content(data: str) -> str | None:
    """
    Extracts text content from a URL or returns raw input if not a URL.
    Uses newspaper3k, then trafilatura, then textfrom.website as fallback.
    """
    if data.startswith(("http://", "https://", "www.")):
        # Normalize URL
        if data.startswith("www."):
            data = "https://" + data

        # Try newspaper3k
        try:
            article = Article(data)
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                print("[INFO] Extracted with newspaper3k.")
                return article.title + " " + article.text
        except Exception as e:
            print(f"[WARN] newspaper3k failed: {e}")

        # Try trafilatura
        try:
            downloaded = trafilatura.fetch_url(data)
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 100:
                print("[INFO] Extracted with trafilatura.")
                return extracted
        except Exception as e:
            print(f"[WARN] trafilatura failed: {e}")

        # Final fallback: textfrom.website
        try:
            tf_url = f"https://textfrom.website/?url={data}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(tf_url, headers=headers, timeout=10)

            if response.ok and len(response.text.strip()) > 100:
                print("[INFO] Extracted with textfrom.website fallback.")
                return response.text
            else:
                print("[WARN] textfrom.website fallback returned too little content.")
        except Exception as e:
            print(f"[ERROR] textfrom.website failed: {e}")

        return None  # All methods failed
    else:
        return data  # Not a URL, treat as raw text


# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive text/URL, process it, and return a fake news prediction.
    Expects JSON input with a 'news_input' field.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    news_input = data.get('news_input')

    if not news_input or not news_input.strip():
        return jsonify({"error": "No text or URL provided in 'news_input' field"}), 400

    print(f"API received input: {news_input[:100]}...") # Log input for debugging

    # Step 1: Get text content using Ishrak's robust scraper
    content_for_prediction = get_text_content(news_input)

    if not content_for_prediction or not content_for_prediction.strip():
        return jsonify({
            "result": "Error",
            "message": "Failed to extract meaningful content from input. Please check URL or provided text.",
            "confidence": None
        }), 400

    # Step 2: Call the team's backend for prediction (TF-IDF model)
    prediction_result_from_team_backend = backend.predict_fake_news_with_confidence(content_for_prediction)

    # Return the results as JSON
    return jsonify(prediction_result_from_team_backend)

# --- Simple HTML Test Page for Flask API (Optional, for direct Flask testing) ---
# This serves your original index.html but connects to the /predict API.
# Save your `index.html` file in a subfolder named `templates` next to this `fake_news_api.py`.
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask API. Make sure 'backend.py' and its 'model.pkl' are in the same directory.")
    print("Flask app will run on http://127.0.0.1:5000/")
    print("Access http://127.0.0.1:5000/ in your browser to test the Flask-only frontend.")
    print("Your Next.js app will connect to this API on port 5000.")
    app.run(debug=True, port=5000)