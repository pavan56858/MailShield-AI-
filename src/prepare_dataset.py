import pandas as pd

INPUT_PATH = "data/raw/phishing_dataset.csv"
OUTPUT_PATH = "data/raw/clean_phishing_dataset.csv"

df = pd.read_csv(INPUT_PATH)

# keep only needed columns
df = df[["text", "label"]]

# drop empty
df.dropna(inplace=True)

# normalize labels
df["label"] = df["label"].astype(str).str.lower().str.strip()

label_map = {
    "phishing": 1,
    "spam": 1,
    "1": 1,
    "legitimate": 0,
    "ham": 0,
    "0": 0
}

df["label"] = df["label"].map(label_map)
df.dropna(inplace=True)

df["label"] = df["label"].astype(int)

# remove empty text
df["text"] = df["text"].astype(str)
df = df[df["text"].str.len() > 20]

print("Final dataset:")
print(df["label"].value_counts())

df.to_csv(OUTPUT_PATH, index=False)

print("✅ Clean dataset saved to:", OUTPUT_PATH)
