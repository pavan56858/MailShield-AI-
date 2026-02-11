from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle, os

# Absolute base project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def create_vectorizer():
    return TfidfVectorizer(max_features=5000, stop_words="english")


def create_scaler():
    return StandardScaler(with_mean=False)


def save_vectorizer(vec):
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)


def save_scaler(scaler):
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def load_vectorizer():
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        return pickle.load(f)


def load_scaler():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)
