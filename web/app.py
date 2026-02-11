from flask import Flask, request, jsonify, render_template
import sys, os, traceback
from datetime import datetime

sys.path.append(os.path.abspath("src"))

from predict import predict_email, load_artefacts

app = Flask(__name__)

print("Loading model...")
MODEL, VECTORIZER, SCALER = load_artefacts()
print("Model loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        email_text = data.get("email_text", "")

        if len(email_text) < 10:
            return jsonify({"success": False, "error": "Email too short"}), 400

        result = predict_email(email_text, MODEL, VECTORIZER, SCALER)

        return jsonify({
            "success": True,
            **result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print("🔥 ERROR in /predict route:")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Backend error. Check terminal."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
