import pickle, os
from preprocessing import clean_text
from feature_engineering import load_vectorizer, load_scaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

LABEL_MAP = {
    0: "Legitimate",
    1: "Phishing"
}

def load_artefacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    vectorizer = load_vectorizer()
    scaler = load_scaler()
    return model, vectorizer, scaler


def predict_email(email_text, model, vectorizer, scaler):
    # Clean text
    text = clean_text(email_text)

    # Vectorize + scale
    X = vectorizer.transform([text])
    X = scaler.transform(X)

    # Predict
    label = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    phishing_prob = float(probs[1])  # probability of class 1 (phishing)

    # Risk logic (more sensitive)
    if phishing_prob >= 0.6:
        prediction = "Phishing"
        risk = "High"
        confidence = phishing_prob * 100
        label_num = 1
    elif phishing_prob >= 0.4:
        prediction = "Phishing"
        risk = "Medium"
        confidence = phishing_prob * 100
        label_num = 1
    else:
        prediction = "Legitimate"
        risk = "Low"
        confidence = (1 - phishing_prob) * 100
        label_num = 0

    return {
        "prediction": prediction,
        "label": label_num,
        "confidence": round(confidence, 2),
        "risk_level": risk,
        "email_preview": email_text[:120]
    }
