import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

MAX_WORDS = 10000
MAX_LENGTH = 200

EMBEDDING_DIM = 128
LSTM_UNITS = 64

EPOCHS = 10
BATCH_SIZE = 32

MODEL_DIR = "models"
ARTIFACT_DIR = "artifacts"
TOKENIZER_PATH = "tokenizer/tokenizer.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# Load processed data
# ============================================================

print("Loading processed data...")

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# MLflow Experiment
# ============================================================

mlflow.set_experiment("Sentiment Analysis TensorFlow")

# ============================================================
# Start MLflow Run
# ============================================================

with mlflow.start_run() as run:

    print("\nMLflow Run ID:")
    print(run.info.run_id)

    # --------------------------------------------------------
    # Log Parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "max_words": MAX_WORDS,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam"
    })


    # ========================================================
    # Build TensorFlow LSTM Model
    # ========================================================

    model = Sequential([
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LENGTH
        ),

        LSTM(LSTM_UNITS),

        Dropout(0.5),

        Dense(32, activation="relu"),

        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])


    # ========================================================
    # Compile Model
    # ========================================================

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    # ========================================================
    # Display Model
    # ========================================================

    model.summary()


    # ========================================================
    # Early Stopping
    # ========================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )


    # ========================================================
    # Train Model
    # ========================================================

    print("\nStarting training...")

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_test,
            y_test
        ),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    # ========================================================
    # Evaluate Model
    # ========================================================

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\nTest Loss:", loss)
    print("Test Accuracy:", accuracy)


    # ========================================================
    # Predictions
    # ========================================================

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int).flatten()


    # ========================================================
    # Classification Metrics
    # ========================================================

    accuracy_value = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    print("\n========== METRICS ==========")

    print("Accuracy :", accuracy_value)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


    # ========================================================
    # Log Metrics to MLflow
    # ========================================================

    mlflow.log_metrics({
        "test_loss": float(loss),
        "test_accuracy": float(accuracy_value),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    })


    # ========================================================
    # Training History Plot
    # ========================================================

    plt.figure()

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    accuracy_plot = os.path.join(
        ARTIFACT_DIR,
        "accuracy.png"
    )

    plt.savefig(accuracy_plot)

    plt.close()


    # ========================================================
    # Loss Plot
    # ========================================================

    plt.figure()

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    loss_plot = os.path.join(
        ARTIFACT_DIR,
        "loss.png"
    )

    plt.savefig(loss_plot)

    plt.close()


    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure()

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    plt.yticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    confusion_plot = os.path.join(
        ARTIFACT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(confusion_plot)

    plt.close()


    # ========================================================
    # Log Artifacts
    # ========================================================

    mlflow.log_artifact(
        accuracy_plot
    )

    mlflow.log_artifact(
        loss_plot
    )

    mlflow.log_artifact(
        confusion_plot
    )


    # ========================================================
    # Save TensorFlow Model Locally
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        "sentiment_model.keras"
    )

    model.save(model_path)

    print("\nModel saved:")
    print(model_path)


    # ========================================================
    # Log Tokenizer
    # ========================================================

    if os.path.exists(TOKENIZER_PATH):

        mlflow.log_artifact(
            TOKENIZER_PATH
        )

        print(
            "Tokenizer logged to MLflow."
        )

    else:

        print(
            "WARNING: tokenizer.pkl not found."
        )


    # ========================================================
    # Log TensorFlow Model to MLflow
    # ========================================================

    print("\nLogging TensorFlow model to MLflow...")

    mlflow.tensorflow.log_model(
        model,
        artifact_path="sentiment_model"
    )


    # ========================================================
    # Register Model
    # ========================================================

    run_id = run.info.run_id

    model_uri = (
        f"runs:/{run_id}/sentiment_model"
    )

    print("\nRegistering model...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="SentimentModel"
    )

    print("\n========== MODEL REGISTERED ==========")

    print(
        "Model Name:",
        registered_model.name
    )

    print(
        "Model Version:",
        registered_model.version
    )

    print(
        "Run ID:",
        run_id
    )


print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")