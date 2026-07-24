\# 💳 AI Credit Card Fraud Detection System



A deep learning-based credit card fraud detection system that analyses transaction data and predicts whether a transaction is normal or potentially fraudulent.



This project implements an end-to-end fraud detection pipeline using \*\*TensorFlow, Keras, SMOTE, Scikit-learn, and Streamlit\*\*.



The system provides fraud probability, risk-level analysis, individual transaction checking, and complete CSV batch prediction.



\---



\## 🚀 Features



\- Deep Neural Network-based fraud detection

\- Correct preprocessing without data leakage

\- Training-only SMOTE class balancing

\- Automatic fraud-threshold selection

\- Fraud probability prediction

\- Individual transaction analysis

\- Complete dataset analysis

\- Fraud-only transaction filtering

\- Risk-level classification

\- Downloadable prediction reports

\- Interactive Streamlit dashboard



\---



\## 🧠 Model Architecture



The fraud detection model is a feed-forward neural network developed using TensorFlow and Keras.



```text

Input Layer

&#x20;   ↓

Dense Layer (128 neurons, ReLU)

&#x20;   ↓

Batch Normalization

&#x20;   ↓

Dropout (30%)

&#x20;   ↓

Dense Layer (64 neurons, ReLU)

&#x20;   ↓

Batch Normalization

&#x20;   ↓

Dropout (25%)

&#x20;   ↓

Dense Layer (32 neurons, ReLU)

&#x20;   ↓

Dropout (15%)

&#x20;   ↓

Output Layer (Sigmoid)

```



The sigmoid output provides a fraud probability between:



\- `0` → Normal transaction

\- `1` → Fraudulent transaction



\---



\## 📊 Dataset



The system uses credit card transaction data.



\### Features



\- `Time`

\- `Amount`

\- `V1` to `V28`



\### Target Column



```text

Class = 0 : Normal Transaction

Class = 1 : Fraudulent Transaction

```



\---



\## 🧹 Data Processing Pipeline



The project applies:



\- Missing-value removal

\- Train, validation, and test split

\- Training-only `StandardScaler` fitting

\- SMOTE only on training data

\- `float32` conversion for TensorFlow

\- Feature-order preservation



These steps prevent data leakage and ensure consistent predictions during deployment.



\---



\## 🎚️ Automatic Threshold Selection



Instead of using only a fixed `0.50` threshold, the system automatically selects the best prediction threshold.



The threshold is selected using:



\- Validation predictions

\- Precision-Recall curve

\- Best F1 Score



The selected threshold is saved and loaded by the Streamlit application.



\---



\## 🚦 Risk Levels



The application provides three risk categories.



\### High Risk 🚨



The transaction probability reaches or exceeds the fraud threshold.



Prediction:



```text

Fraud

```



\### Needs Review ⚠️



The transaction probability is close to the fraud threshold and requires manual checking.



\### Low Risk ✅



The transaction has a low fraud probability.



Prediction:



```text

Normal

```



\---



\## 🌐 Streamlit Application



The web application provides two main analysis modes.



\### Individual Transaction Analysis



Users can enter a transaction line number and view:



\- Transaction amount

\- Transaction time

\- All transaction features

\- Fraud probability

\- Risk level

\- Prediction result

\- Actual label, when available



\### Complete Dataset Analysis



The application analyses complete CSV files and generates:



\- Complete prediction report

\- Fraud-only report

\- Fraud-probability ranking

\- Evaluation metrics



\---



\## 📥 Download Reports



The system provides downloadable CSV reports.



\### Complete Report



Contains:



\- Line Number

\- Transaction Features

\- Fraud Probability

\- Predicted Class

\- Prediction

\- Risk Level



\### Fraud Report



Contains only transactions predicted as fraudulent.



\---



\## ⚙️ Technologies Used



\- Python

\- TensorFlow

\- Keras

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- Imbalanced-learn

\- SMOTE

\- Joblib

\- Matplotlib



\---



\## 📁 Project Structure



```text

AI-Credit-Card-Fraud-Detection-System/

│

├── streamlit\_app.py

├── train.py

├── requirements.txt

├── README.md

├── LICENSE

│

├── Screenshot/

│

└── fraud\_detection\_artifacts/

&#x20;   ├── fraud\_detection\_model.keras

&#x20;   ├── scaler.pkl

&#x20;   ├── preprocessing\_metadata.pkl

&#x20;   ├── prediction\_threshold.pkl

&#x20;   ├── confusion\_matrix.png

&#x20;   └── roc\_curve.png

```



\---



\## 📦 Installation



Clone the repository:



```bash

git clone https://github.com/Anisa960/AI-Credit-Card-Fraud-Detection-System.git

```



Move into the project folder:



```bash

cd AI-Credit-Card-Fraud-Detection-System

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 🏋️ Train the Model



Run:



```bash

python train.py

```



The training process will:



1\. Load the dataset

2\. Validate required features

3\. Preprocess the data

4\. Split the dataset

5\. Fit the scaler only on training data

6\. Apply SMOTE only to training data

7\. Train the neural network

8\. Save the best model

9\. Select the optimal threshold

10\. Generate evaluation graphs



\---



\## ▶️ Run the Streamlit Application



Run:



```bash

streamlit run streamlit\_app.py

```



Open the following address in your browser:



```text

http://localhost:8501

```



\---



\## 📈 Model Evaluation



The model evaluates:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- ROC AUC

\- Classification Report

\- Confusion Matrix

\- ROC Curve



\---



\## 💾 Saved Artifacts



The training script generates the following files:



```text

fraud\_detection\_artifacts/

├── fraud\_detection\_model.keras

├── scaler.pkl

├── preprocessing\_metadata.pkl

├── prediction\_threshold.pkl

├── confusion\_matrix.png

└── roc\_curve.png

```



\### Artifact Purpose



\- `fraud\_detection\_model.keras` — trained TensorFlow model

\- `scaler.pkl` — scaler fitted only on training data

\- `preprocessing\_metadata.pkl` — feature order and scaled-column information

\- `prediction\_threshold.pkl` — automatically selected fraud threshold

\- `confusion\_matrix.png` — confusion-matrix visualisation

\- `roc\_curve.png` — ROC-curve visualisation



\---



\## 🔎 Individual Transaction Testing



The Streamlit application allows users to select a transaction by line number.



For example:



```text

Line Number: 542

```



The system displays:



\- Transaction details

\- Fraud probability

\- Risk level

\- Threshold used

\- Final prediction

\- Actual dataset class, when available

\- Whether the prediction matches the actual class



The application line number starts from the first transaction after the CSV header.



```text

App Line 1 = First Transaction Row

Physical CSV Line 2 = First Transaction Row

```



\---



\## 🛠️ Problems Corrected During Development



The following issues were identified and fixed:



\### Scaler Data Leakage



The scaler was previously fitted before the dataset split. It is now fitted only on the training data.



\### Duplicate Scaler Saving



The duplicate scaler-saving operation was removed.



\### SMOTE and Class-Weight Conflict



Class weights were removed because SMOTE already balances the training dataset.



\### Correct Saved-Model Testing



The saved scaler is now applied before predictions are made using the saved model.



\### Feature-Order Consistency



The original training feature order is saved and reused during Streamlit predictions.



\### Validation-Based Threshold



The fixed `0.50` threshold was replaced with an automatically selected threshold based on the best validation F1 score.



\### Best Model Checkpoint



Only the best-performing model according to validation loss is saved.



\### TensorFlow Data Compatibility



All model input arrays are converted to `float32`.



\### Safe Metric Calculation



Evaluation metrics use `zero\_division=0` to prevent division warnings.



\---



\## ⚠️ Limitations



\- Dataset features `V1` to `V28` are anonymised

\- The model does not contain customer identity information

\- New fraud patterns may require retraining

\- Prediction is a decision-support result, not final proof

\- The current project does not include real-time banking integration

\- The current project does not include SHAP-based explainability



\---



\## 🔮 Future Improvements



\- SHAP Explainable AI

\- Real-time fraud-detection API

\- Database integration

\- User authentication

\- Cloud deployment

\- Model monitoring

\- Automated retraining

\- Email or SMS fraud alerts

\- Merchant and device risk analysis

\- Secure banking-system integration



\---



\## 📸 Screenshots



The repository can include screenshots of:



\- Main dashboard

\- Individual transaction analysis

\- Fraud probability result

\- Fraud transaction report

\- Complete dataset analysis

\- Downloadable results



Store screenshots inside:



```text

Screenshot/

```



\---



\## 📜 License



This project is licensed under the MIT License.



\---



\## 👩‍💻 Author



\*\*Anisa Ramzan\*\*



AI / Machine Learning Developer



