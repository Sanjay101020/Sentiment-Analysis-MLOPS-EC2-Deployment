import os
import pickle
import numpy as np
import mlflow
import mlflow.tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Configuration
# ============================================================

MAX_LENGTH = 200

MODEL_NAME = "SentimentModel"

TOKENIZER_PATH = "tokenizer/tokenizer.pkl"


# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.")


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# ============================================================
# Load Model from MLflow
# ============================================================

def load_model():
    

    print("Loading model from MLflow...")

    # Load latest registered model version

    model_uri = f"models:/{MODEL_NAME}/latest"

    try:

        model = mlflow.tensorflow.load_model(
            model_uri
        )

        print("MLflow model loaded successfully.")

        return model

    except Exception as e:

        print("Could not load latest registered model.")
        print("Error:", e)

        raise


model = load_model()


# ============================================================
# Prediction Function
# ============================================================

def predict_sentiment(review):

    if not review or not review.strip():

        raise ValueError(
            "Review cannot be empty."
        )


    # --------------------------------------------------------
    # Convert text to lowercase
    # --------------------------------------------------------

    review = review.lower()


    # --------------------------------------------------------
    # Convert text to sequence
    # --------------------------------------------------------

    sequence = tokenizer.texts_to_sequences(
        [review]
    )


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if probability >= 0.5:

        prediction = "Positive"

        confidence = probability * 100

    else:

        prediction = "Negative"

        confidence = (1 - probability) * 100


    return prediction, confidence


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("Movie Review Sentiment Prediction")
    print("======================================")

    while True:

        review = input(
            "\nEnter movie review "
            "(type 'exit' to stop): "
        )

        if review.lower() == "exit":
            break

        try:

            prediction, confidence = (
                predict_sentiment(review)
            )

            print()
            print("Review     :", review)
            print("Prediction :", prediction)
            print(
                "Confidence : "
                f"{confidence:.2f}%"
            )

        except Exception as e:

            print(
                "Prediction Error:",
                e
            )