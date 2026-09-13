import os
import re
import pickle

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# Configuration
# ============================================================


MAX_LENGTH = 200

TOKENIZER_PATH = "tokenizer/tokenizer.pkl"


# ============================================================
# Load Tokenizer
# ============================================================

if os.getenv("TESTING") == "true":

    tokenizer = None

else:

    print("Loading tokenizer...")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    print("Tokenizer loaded successfully.")

# ============================================================
# Load MLflow Model
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001"
)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "sentiment_model.keras"
)
if os.getenv("TESTING") == "true":
    model = None
else:
    model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully:", MODEL_PATH)


MODEL_NAME = "SentimentModel"

# Change this when you want to deploy another registered version
MODEL_VERSION = "1"

MODEL_URI = (
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

# print("Loading MLflow model...")
# print("Model URI:", MODEL_URI)

# model = mlflow.tensorflow.load_model(
#     MODEL_URI
# )

# print("Model loaded successfully.")


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Prediction
# ============================================================

def predict_sentiment(review):

    review = clean_text(review)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [review]
    )

    # Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Prediction
    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Classification
    if probability >= 0.5:

        prediction = "Positive"

        confidence = probability * 100

    else:

        prediction = "Negative"

        confidence = (1 - probability) * 100

    return prediction, confidence


# ============================================================
# Home Page
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Prediction Route
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    review = request.form.get(
        "review",
        ""
    )

    if not review.strip():

        return render_template(
            "index.html",
            error="Please enter a movie review."
        )

    try:

        prediction, confidence = (
            predict_sentiment(review)
        )

        return render_template(
            "index.html",

            review=review,

            prediction=prediction,

            confidence=f"{confidence:.2f}"
        )

    except Exception as e:

        return render_template(
            "index.html",

            error=f"Prediction error: {str(e)}"
        )


# ============================================================
# Health Check
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
