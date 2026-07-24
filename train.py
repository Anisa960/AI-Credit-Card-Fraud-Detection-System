# ============================================================
# Credit Card Fraud Detection — Corrected Complete Version
# ============================================================

import os

# Set these before importing TensorFlow
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

from tensorflow.keras import layers, callbacks, models
from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "creditcard.csv"
ARTIFACT_DIR = "fraud_detection_artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "fraud_detection_model.keras"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.pkl"
)

METADATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "preprocessing_metadata.pkl"
)

THRESHOLD_PATH = os.path.join(
    ARTIFACT_DIR,
    "prediction_threshold.pkl"
)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load Dataset
# ============================================================

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "Time",
        "Amount",
        "Class"
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.dropna().copy()

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError(
            "Class column must contain only 0 and 1."
        )

    print("\nDataset loaded successfully")
    print("Dataset shape:", df.shape)

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentage:")
    print(
        df["Class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )

    return df


# ============================================================
# Preprocessing
# ============================================================

def preprocess_data(df):

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].astype("int32").copy()

    feature_columns = X.columns.tolist()
    scaled_columns = ["Time", "Amount"]

    # --------------------------------------------------------
    # Split before scaling to prevent data leakage
    # --------------------------------------------------------

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=SEED,
        stratify=y
    )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train
    )

    # --------------------------------------------------------
    # Fit scaler only on training data
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_train_raw[scaled_columns]
    )

    def apply_scaling(data):

        data = data.copy()

        data[scaled_columns] = scaler.transform(
            data[scaled_columns]
        )

        return data

    X_train = apply_scaling(X_train_raw)
    X_val = apply_scaling(X_val_raw)
    X_test = apply_scaling(X_test_raw)

    # --------------------------------------------------------
    # Save scaler once
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    preprocessing_metadata = {
        "feature_columns": feature_columns,
        "scaled_columns": scaled_columns
    }

    joblib.dump(
        preprocessing_metadata,
        METADATA_PATH
    )

    print("\nScaler saved successfully")
    print("Scaler fitted only on training data")

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # Apply SMOTE only to training data
    # --------------------------------------------------------

    smote = SMOTE(
        random_state=SEED
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Convert to float32 for TensorFlow
    X_train_resampled = np.asarray(
        X_train_resampled,
        dtype=np.float32
    )

    X_val = np.asarray(
        X_val,
        dtype=np.float32
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    y_train_resampled = np.asarray(
        y_train_resampled,
        dtype=np.float32
    )

    y_val = np.asarray(
        y_val,
        dtype=np.float32
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int32
    )

    return {
        "X_train": X_train_resampled,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_resampled,
        "y_val": y_val,
        "y_test": y_test,
        "X_test_raw": X_test_raw,
        "feature_columns": feature_columns,
        "scaled_columns": scaled_columns
    }


# ============================================================
# Build Neural Network
# ============================================================

def build_model(input_features):

    model = models.Sequential([
        layers.Input(shape=(input_features,)),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.30),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.25),

        layers.Dense(
            32,
            activation="relu"
        ),

        layers.Dropout(0.15),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


# ============================================================
# Find Best Threshold Using Validation Dataset
# ============================================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # Precision and recall contain one extra value
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
    ) / (
        precision + recall + 1e-8
    )

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    return float(best_threshold), float(best_f1)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test, threshold):

    y_probability = model.predict(
        X_test,
        batch_size=1024,
        verbose=0
    ).flatten()

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n" + "=" * 45)
    print("TEST RESULTS")
    print("=" * 45)

    print(f"Threshold : {threshold:.6f}")
    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")
    print(f"ROC AUC   : {roc_auc:.6f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=[
                "Normal",
                "Fraud"
            ],
            zero_division=0
        )
    )

    return y_probability, y_prediction


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(y_test, y_prediction):

    matrix = confusion_matrix(
        y_test,
        y_prediction
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Normal", "Fraud"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Fraud"]
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):

            plt.text(
                column,
                row,
                matrix[row, column],
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        ARTIFACT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Confusion matrix saved: "
        f"{confusion_matrix_path}"
    )


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(y_test, y_probability):

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_probability
    )

    auc_score = roc_auc_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC AUC = {auc_score:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()

    roc_curve_path = os.path.join(
        ARTIFACT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        roc_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"ROC curve saved: {roc_curve_path}"
    )


# ============================================================
# Test Saved Model and Saved Scaler
# ============================================================

def test_saved_artifacts(X_test_raw, y_test):

    loaded_model = load_model(
        MODEL_PATH
    )

    loaded_scaler = joblib.load(
        SCALER_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    loaded_threshold = joblib.load(
        THRESHOLD_PATH
    )

    feature_columns = metadata["feature_columns"]
    scaled_columns = metadata["scaled_columns"]

    # Take raw, unscaled transactions
    sample = X_test_raw.iloc[:5].copy()

    # Ensure correct feature order
    sample = sample[feature_columns]

    # Apply saved scaler
    sample[scaled_columns] = loaded_scaler.transform(
        sample[scaled_columns]
    )

    sample_array = sample.to_numpy(
        dtype=np.float32
    )

    probabilities = loaded_model.predict(
        sample_array,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= loaded_threshold
    ).astype(int)

    actual_values = y_test[:5]

    print("\n" + "=" * 45)
    print("SAVED MODEL SAMPLE PREDICTIONS")
    print("=" * 45)

    for index in range(len(predictions)):

        predicted_label = (
            "Fraud"
            if predictions[index] == 1
            else "Normal"
        )

        actual_label = (
            "Fraud"
            if actual_values[index] == 1
            else "Normal"
        )

        print(
            f"Transaction {index + 1}: "
            f"Probability={probabilities[index]:.6f}, "
            f"Prediction={predicted_label}, "
            f"Actual={actual_label}"
        )


# ============================================================
# Main Program
# ============================================================

def main():

    df = load_data(DATA_PATH)

    data = preprocess_data(df)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Testing shape:", X_test.shape)

    model = build_model(
        input_features=X_train.shape[1]
    )

    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    reduce_learning_rate = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    print("\nModel training started...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=30,
        batch_size=256,
        callbacks=[
            early_stopping,
            reduce_learning_rate,
            model_checkpoint
        ],
        verbose=1
    )

    # Load best saved model
    best_model = load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Select threshold from validation data
    # --------------------------------------------------------

    validation_probabilities = best_model.predict(
        X_val,
        batch_size=1024,
        verbose=0
    ).flatten()

    best_threshold, validation_f1 = find_best_threshold(
        y_val,
        validation_probabilities
    )

    joblib.dump(
        best_threshold,
        THRESHOLD_PATH
    )

    print("\nBest validation threshold:")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"Validation F1: {validation_f1:.6f}")

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    y_probability, y_prediction = evaluate_model(
        best_model,
        X_test,
        y_test,
        best_threshold
    )

    plot_confusion_matrix(
        y_test,
        y_prediction
    )

    plot_roc_curve(
        y_test,
        y_probability
    )

    test_saved_artifacts(
        data["X_test_raw"],
        y_test
    )

    print("\n" + "=" * 45)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 45)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print(f"Threshold saved: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()