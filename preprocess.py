import os
import re
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/imdb.csv"

TOKENIZER_DIR = "tokenizer"
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

MAX_WORDS = 10000
MAX_LENGTH = 200


# ============================================================
# Create directories
# ============================================================

os.makedirs(TOKENIZER_DIR, exist_ok=True)


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# Validate columns
# ============================================================

required_columns = ["review", "sentiment"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}. "
            f"Required columns are: {required_columns}"
        )


# ============================================================
# Remove missing values
# ============================================================

df = df.dropna(subset=["review", "sentiment"])


# ============================================================
# Clean reviews
# ============================================================

print("Cleaning text...")

df["review"] = df["review"].apply(clean_text)


# ============================================================
# Convert sentiment to numeric
# ============================================================

def convert_sentiment(value):

    value = str(value).lower().strip()

    if value in ["positive", "pos", "1"]:
        return 1

    elif value in ["negative", "neg", "0"]:
        return 0

    else:
        raise ValueError(
            f"Unknown sentiment value: {value}"
        )


df["sentiment"] = df["sentiment"].apply(convert_sentiment)


# ============================================================
# Features and labels
# ============================================================

X = df["review"].values
y = df["sentiment"].values


# ============================================================
# Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# Tokenizer
# ============================================================

print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


# ============================================================
# Convert text to sequences
# ============================================================

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)


# ============================================================
# Padding
# ============================================================

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# Save tokenizer
# ============================================================

with open(TOKENIZER_PATH, "wb") as file:
    pickle.dump(tokenizer, file)


print(f"Tokenizer saved to: {TOKENIZER_PATH}")


# ============================================================
# Save processed data
# ============================================================

np.save("X_train.npy", X_train_padded)
np.save("X_test.npy", X_test_padded)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)


print("Processed data saved successfully.")

print()
print("========== PREPROCESSING COMPLETE ==========")
print("X_train:", X_train_padded.shape)
print("X_test :", X_test_padded.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)