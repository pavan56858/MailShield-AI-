import pandas as pd
import os, pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import clean_text
from feature_engineering import create_vectorizer, create_scaler, save_vectorizer, save_scaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = "data/raw/test_dataset.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns)

# Keep only needed columns
df = df[["text", "label"]]

# Drop missing
df.dropna(inplace=True)

# Normalize labels
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Print unique labels for debug
print("Unique labels:", df["label"].unique()[:10])

# Map labels (handle multiple possibilities)
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

# Drop rows where label is still NaN
df.dropna(inplace=True)

df["label"] = df["label"].astype(int)

print("Label distribution:")
print(df["label"].value_counts())

# Clean text
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# Remove empty cleaned text
df = df[df["clean_text"].str.len() > 0]

print("Sample cleaned text:")
print(df["clean_text"].head())

X = df["clean_text"]
y = df["label"]

# Vectorize
vectorizer = create_vectorizer()
X_vec = vectorizer.fit_transform(X)

# Scale
scaler = create_scaler()
X_scaled = scaler.fit_transform(X_vec)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

save_vectorizer(vectorizer)
save_scaler(scaler)

print("✅ Model trained and saved successfully")
