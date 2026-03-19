from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

# ── Load model and tokenizer ───────────────────────────────────────────
MODEL_PATH = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=3, local_files_only=True
).to(device)
model.eval()

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ── Same Arabizi cleaner from preprocessing ────────────────────────────
def clean_arabizi(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = text.lower()
    arabizi_map = {
        '3': 'ع', '2': 'ء', '7': 'ح',
        '5': 'خ', '6': 'ط', '8': 'غ',
        '9': 'ق', '4': 'ذ'
    }
    for digit, letter in arabizi_map.items():
        text = re.sub(
            rf'(?<=[a-zA-Z\u0600-\u06FF]){digit}|{digit}(?=[a-zA-Z\u0600-\u06FF])',
            letter, text
        )
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "mBERT Arabizi Sentiment"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Request must include 'text' field"}), 400

    text = data["text"]
    cleaned = clean_arabizi(text)

    inputs = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    pred  = logits.argmax(-1).item()

    return jsonify({
        "input":      text,
        "cleaned":    cleaned,
        "sentiment":  LABELS[pred],
        "confidence": round(max(probs), 4),
        "scores": {
            "negative": round(probs[0], 4),
            "neutral":  round(probs[1], 4),
            "positive": round(probs[2], 4)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)