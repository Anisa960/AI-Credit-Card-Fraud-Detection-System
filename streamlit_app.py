import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .normal-result {
        background-color: rgba(0, 180, 80, 0.12);
        border: 1px solid rgba(0, 180, 80, 0.45);
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
    }

    .fraud-result {
        background-color: rgba(255, 0, 0, 0.12);
        border: 1px solid rgba(255, 0, 0, 0.45);
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
    }

    .review-result {
        background-color: rgba(255, 165, 0, 0.12);
        border: 1px solid rgba(255, 165, 0, 0.45);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        AI-based transaction risk analysis system
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Load Model Assets
# ============================================================

@st.cache_resource
def load_assets():

    required_files = {
        "Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Metadata": METADATA_PATH,
        "Threshold": THRESHOLD_PATH
    }

    missing_files = []

    for file_name, file_path in required_files.items():

        if not os.path.exists(file_path):
            missing_files.append(
                f"{file_name}: {file_path}"
            )

    if missing_files:

        raise FileNotFoundError(
            "The following required files are missing:\n\n"
            + "\n".join(missing_files)
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    loaded_scaler = joblib.load(
        SCALER_PATH
    )

    loaded_metadata = joblib.load(
        METADATA_PATH
    )

    loaded_threshold = float(
        joblib.load(THRESHOLD_PATH)
    )

    return (
        loaded_model,
        loaded_scaler,
        loaded_metadata,
        loaded_threshold
    )


try:

    model, scaler, metadata, saved_threshold = load_assets()

except Exception as error:

    st.error(
        f"Model assets could not be loaded:\n\n{error}"
    )

    st.stop()


feature_columns = metadata.get(
    "feature_columns"
)

scaled_columns = metadata.get(
    "scaled_columns",
    ["Time", "Amount"]
)

if not feature_columns:

    st.error(
        "Feature column information is missing from metadata."
    )

    st.stop()


# ============================================================
# Sidebar Settings
# ============================================================

st.sidebar.header("⚙️ System Settings")

threshold_mode = st.sidebar.radio(
    "Fraud decision threshold",
    options=[
        "Use trained threshold",
        "Use manual threshold"
    ]
)

if threshold_mode == "Use manual threshold":

    active_threshold = st.sidebar.slider(
        "Select threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01
    )

else:

    active_threshold = saved_threshold


st.sidebar.metric(
    "Active Threshold",
    f"{active_threshold:.6f}"
)

st.sidebar.caption(
    "A transaction is classified as fraud when its predicted "
    "probability is equal to or greater than this threshold."
)


# ============================================================
# Data Preparation Functions
# ============================================================

def validate_dataframe(dataframe):

    if dataframe.empty:
        raise ValueError(
            "The uploaded CSV file contains no transactions."
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            + ", ".join(missing_columns)
        )


def prepare_features(dataframe):

    validate_dataframe(dataframe)

    features = dataframe[
        feature_columns
    ].copy()

    for column in feature_columns:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    invalid_columns = features.columns[
        features.isnull().any()
    ].tolist()

    if invalid_columns:
        raise ValueError(
            "Missing or invalid numeric values found in:\n"
            + ", ".join(invalid_columns)
        )

    missing_scaled_columns = [
        column
        for column in scaled_columns
        if column not in features.columns
    ]

    if missing_scaled_columns:
        raise ValueError(
            "Columns required by the scaler are missing:\n"
            + ", ".join(missing_scaled_columns)
        )

    features[scaled_columns] = scaler.transform(
        features[scaled_columns]
    )

    return features.to_numpy(
        dtype=np.float32
    )


def predict_transactions(dataframe):

    prepared_data = prepare_features(
        dataframe
    )

    probabilities = model.predict(
        prepared_data,
        batch_size=1024,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= active_threshold
    ).astype(int)

    return probabilities, predictions


def get_risk_level(probability, prediction):

    if prediction == 1:
        return "High Risk"

    if probability >= active_threshold * 0.70:
        return "Needs Review"

    return "Low Risk"


# ============================================================
# CSV Upload
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload transaction CSV file",
    type=["csv"],
    help=(
        "Upload the creditcard.csv dataset or another CSV file "
        "with the same transaction features."
    )
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file to analyze its transactions."
    )

    st.stop()


try:

    dataframe = pd.read_csv(
        uploaded_file
    )

    validate_dataframe(
        dataframe
    )

except Exception as error:

    st.error(
        f"CSV file could not be processed:\n\n{error}"
    )

    st.stop()


# Add user-friendly line numbers
display_dataframe = dataframe.copy()

display_dataframe.insert(
    0,
    "Line Number",
    np.arange(1, len(display_dataframe) + 1)
)


# ============================================================
# Dataset Overview
# ============================================================

st.success(
    "✅ Model, scaler and dataset loaded successfully."
)

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:

    st.metric(
        "Total Transactions",
        f"{len(dataframe):,}"
    )

with overview_col2:

    st.metric(
        "Transaction Features",
        len(feature_columns)
    )

with overview_col3:

    st.metric(
        "Fraud Threshold",
        f"{active_threshold:.6f}"
    )


st.subheader("📊 Dataset Preview")

st.caption(
    "Line Number 1 means the first transaction after the CSV header."
)

st.dataframe(
    display_dataframe.head(50),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Tabs
# ============================================================

single_tab, batch_tab = st.tabs(
    [
        "🔎 Check One Transaction",
        "📊 Analyze Complete Dataset"
    ]
)


# ============================================================
# Single Transaction Analysis
# ============================================================

with single_tab:

    st.subheader(
        "Select a Transaction Line"
    )

    selected_line = st.number_input(
        "Enter line number",
        min_value=1,
        max_value=len(dataframe),
        value=1,
        step=1,
        help=(
            "For example, enter 1 to analyze the first transaction "
            "or enter 100 to analyze transaction number 100."
        )
    )

    selected_index = int(selected_line) - 1

    selected_transaction = dataframe.iloc[
        [selected_index]
    ].copy()

    transaction_data = dataframe.iloc[
        selected_index
    ]


    # --------------------------------------------------------
    # Show basic transaction information
    # --------------------------------------------------------

    st.markdown(
        f"### Transaction Line #{int(selected_line)}"
    )

    information_col1, information_col2, information_col3 = st.columns(3)

    with information_col1:

        if "Amount" in transaction_data.index:

            st.metric(
                "Transaction Amount",
                f"{float(transaction_data['Amount']):,.2f}"
            )

        else:

            st.metric(
                "Transaction Amount",
                "Not available"
            )

    with information_col2:

        if "Time" in transaction_data.index:

            st.metric(
                "Transaction Time",
                f"{float(transaction_data['Time']):,.2f}"
            )

        else:

            st.metric(
                "Transaction Time",
                "Not available"
            )

    with information_col3:

        st.metric(
            "CSV Line Number",
            int(selected_line)
        )


    with st.expander(
        "View all transaction features"
    ):

        feature_view = pd.DataFrame({
            "Feature": feature_columns,
            "Value": [
                transaction_data[column]
                for column in feature_columns
            ]
        })

        st.dataframe(
            feature_view,
            use_container_width=True,
            hide_index=True
        )


    analyze_transaction_button = st.button(
        "🔍 Analyze Selected Transaction",
        type="primary",
        use_container_width=True
    )


    if analyze_transaction_button:

        try:

            with st.spinner(
                "AI is checking the selected transaction..."
            ):

                probability, prediction = predict_transactions(
                    selected_transaction
                )

                fraud_probability = float(
                    probability[0]
                )

                predicted_class = int(
                    prediction[0]
                )

                risk_level = get_risk_level(
                    fraud_probability,
                    predicted_class
                )

        except Exception as error:

            st.error(
                f"Transaction analysis failed:\n\n{error}"
            )

            st.stop()


        st.divider()

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.4f}%"
            )

        with result_col2:

            st.metric(
                "Risk Level",
                risk_level
            )

        with result_col3:

            st.metric(
                "Threshold Used",
                f"{active_threshold:.6f}"
            )


        st.progress(
            min(
                max(fraud_probability, 0.0),
                1.0
            ),
            text=(
                f"AI Fraud Risk Score: "
                f"{fraud_probability * 100:.4f}%"
            )
        )


        if predicted_class == 1:

            st.markdown(
                """
                <div class="fraud-result">
                    🚨 Fraudulent Transaction Detected
                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "This transaction crossed the fraud threshold and "
                "should be reviewed or temporarily blocked."
            )

        else:

            st.markdown(
                """
                <div class="normal-result">
                    ✅ Normal Transaction
                </div>
                """,
                unsafe_allow_html=True
            )

            if risk_level == "Needs Review":

                st.warning(
                    "The transaction is classified as normal, but its "
                    "risk score is close to the fraud threshold."
                )

            else:

                st.success(
                    "The transaction is below the fraud threshold."
                )


        # ----------------------------------------------------
        # Show actual class when available
        # ----------------------------------------------------

        if "Class" in dataframe.columns:

            actual_value = pd.to_numeric(
                pd.Series(
                    [transaction_data["Class"]]
                ),
                errors="coerce"
            ).iloc[0]

            if pd.notna(actual_value) and int(actual_value) in [0, 1]:

                actual_class = int(
                    actual_value
                )

                actual_label = (
                    "Fraud 🚨"
                    if actual_class == 1
                    else "Normal ✅"
                )

                prediction_status = (
                    "Correct Prediction ✅"
                    if actual_class == predicted_class
                    else "Incorrect Prediction ❌"
                )

                actual_col1, actual_col2 = st.columns(2)

                with actual_col1:

                    st.metric(
                        "Actual Dataset Label",
                        actual_label
                    )

                with actual_col2:

                    st.metric(
                        "Model Comparison",
                        prediction_status
                    )


# ============================================================
# Complete Dataset Analysis
# ============================================================

with batch_tab:

    st.subheader(
        "Analyze All Uploaded Transactions"
    )

    st.write(
        "This option checks every transaction in the uploaded CSV "
        "and creates a downloadable fraud report."
    )

    analyze_all_button = st.button(
        "🚀 Analyze Complete Dataset",
        type="primary",
        use_container_width=True
    )


    if analyze_all_button:

        try:

            with st.spinner(
                "AI is analyzing all transactions..."
            ):

                probabilities, predictions = predict_transactions(
                    dataframe
                )

                result = dataframe.copy()

                result.insert(
                    0,
                    "Line Number",
                    np.arange(1, len(result) + 1)
                )

                result["Fraud Probability"] = (
                    probabilities * 100
                ).round(6)

                result["Predicted Class"] = predictions

                result["Prediction"] = np.where(
                    predictions == 1,
                    "Fraud 🚨",
                    "Normal ✅"
                )

                result["Risk Level"] = [
                    get_risk_level(
                        float(probability),
                        int(prediction)
                    )
                    for probability, prediction
                    in zip(probabilities, predictions)
                ]

        except Exception as error:

            st.error(
                f"Complete dataset analysis failed:\n\n{error}"
            )

            st.stop()


        fraud_count = int(
            predictions.sum()
        )

        normal_count = int(
            len(predictions) - fraud_count
        )

        fraud_percentage = (
            fraud_count / len(predictions)
        ) * 100


        batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)

        with batch_col1:

            st.metric(
                "Total Transactions",
                f"{len(result):,}"
            )

        with batch_col2:

            st.metric(
                "Fraud Detected",
                f"{fraud_count:,}"
            )

        with batch_col3:

            st.metric(
                "Normal Transactions",
                f"{normal_count:,}"
            )

        with batch_col4:

            st.metric(
                "Fraud Percentage",
                f"{fraud_percentage:.4f}%"
            )


        # ----------------------------------------------------
        # Evaluation when actual Class is available
        # ----------------------------------------------------

        if "Class" in dataframe.columns:

            actual_classes = pd.to_numeric(
                dataframe["Class"],
                errors="coerce"
            )

            valid_actual_classes = (
                not actual_classes.isnull().any()
                and set(actual_classes.unique()).issubset({0, 1})
            )

            if valid_actual_classes:

                actual_classes = actual_classes.astype(int)

                accuracy = accuracy_score(
                    actual_classes,
                    predictions
                )

                precision = precision_score(
                    actual_classes,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    actual_classes,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    actual_classes,
                    predictions,
                    zero_division=0
                )

                st.subheader(
                    "📈 Model Performance on Uploaded Data"
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}"
                    )

                with metric_col2:

                    st.metric(
                        "Precision",
                        f"{precision:.4f}"
                    )

                with metric_col3:

                    st.metric(
                        "Recall",
                        f"{recall:.4f}"
                    )

                with metric_col4:

                    st.metric(
                        "F1 Score",
                        f"{f1:.4f}"
                    )


        # ----------------------------------------------------
        # Complete results
        # ----------------------------------------------------

        st.subheader(
            "📋 Complete Prediction Report"
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True,
            height=500
        )


        # ----------------------------------------------------
        # Fraud-only results
        # ----------------------------------------------------

        fraud_transactions = result[
            result["Predicted Class"] == 1
        ].copy()

        fraud_transactions = fraud_transactions.sort_values(
            by="Fraud Probability",
            ascending=False
        )


        st.subheader(
            "🚨 Fraudulent Transactions"
        )

        if fraud_transactions.empty:

            st.success(
                "No fraudulent transaction was detected."
            )

        else:

            st.warning(
                f"{len(fraud_transactions):,} transactions require review."
            )

            st.dataframe(
                fraud_transactions,
                use_container_width=True,
                hide_index=True,
                height=400
            )


        # ----------------------------------------------------
        # Downloads
        # ----------------------------------------------------

        st.subheader(
            "⬇️ Download Reports"
        )

        complete_report_csv = result.to_csv(
            index=False
        ).encode("utf-8")

        fraud_report_csv = fraud_transactions.to_csv(
            index=False
        ).encode("utf-8")


        download_col1, download_col2 = st.columns(2)

        with download_col1:

            st.download_button(
                label="⬇️ Download Complete Report",
                data=complete_report_csv,
                file_name="complete_fraud_detection_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download_col2:

            st.download_button(
                label="🚨 Download Fraud Report",
                data=fraud_report_csv,
                file_name="fraud_transactions_report.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=fraud_transactions.empty
            )