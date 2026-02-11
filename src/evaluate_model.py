import pickle
import pandas as pd
from preprocessing import clean_text
from feature_engineering import load_vectorizer, load_scaler
from sklearn.metrics import classification_report, accuracy_score

# Load model and artefacts
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = load_vectorizer()
scaler = load_scaler()

# Load dataset
df = pd.read_csv("data/raw/test_dataset.csv")

# Keep only needed columns
df = df[["text", "label"]]
df.dropna(inplace=True)

# Normalize labels
df["label"] = df["label"].astype(str).str.lower().str.strip()

label_map = {
    "phishing": 1,
    "spam": 1,
    "1": 1,
    1: 1,
    "legitimate": 0,
    "ham": 0,
    "0": 0,
    0: 0
}

df["label"] = df["label"].map(label_map)
df.dropna(inplace=True)
df["label"] = df["label"].astype(int)

# Clean text
df["clean_text"] = df["text"].astype(str).apply(clean_text)
df = df[df["clean_text"].str.len() > 0]

# Transform
X = scaler.transform(vectorizer.transform(df["clean_text"]))
y = df["label"]

# Predict
y_pred = model.predict(X)

# Evaluate
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
