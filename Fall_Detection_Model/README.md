# Fall Detection Model

A machine learning pipeline for detecting **falls using raw accelerometer data (x, y, z)**, designed for integration with a wearable health monitoring system.

## 🚀 Overview

This project trains a binary classifier (Fall vs Non-Fall) using the [Fall Detection Accelerometer Dataset](https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data/data). Each raw `(AccelerationX, AccelerationY, AccelerationZ)` reading is classified directly. The primary model is **XGBoost**, with Random Forest, SVM, and Logistic Regression included for comparison.

## 📂 Project Structure

```
Fall_Detection_Model/
├── data/                        # Dataset root (activity subfolders)
├── models/                      # Trained model outputs (auto-generated)
├── src/
│   ├── data_loader.py           # Load CSVs, flatten to raw x,y,z rows
│   ├── preprocessing.py         # Clean & standardize
│   ├── feature_engineering.py   # Pass-through (raw features)
│   ├── model_training.py        # Train classifiers + tuning
│   ├── evaluation.py            # Metrics, plots, comparisons
│   └── export_model.py          # Save model & pipeline artifacts
├── main.py                      # Full pipeline entry point
├── requirements.txt
├── .gitignore                   # Git ignore file
└── README.md
```

## ⚙️ Setup Instructions

1. **Clone the repository and navigate to the directory.**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data/data) and extract it into the `data/` directory. The dataset should contain **6 activity folders** (`freeFall`, `runFall`, `walkFall`, `downSit`, `runSit`, `walkSit`), each with semicolon-separated CSV files.

## 💻 Usage

Run the main pipeline using the `main.py` entry point:

```bash
python main.py              # Full pipeline with hyperparameter tuning
python main.py --no-tune    # Quick run (skip tuning)
python main.py --no-tune --no-smote --test-size 0.3  # Custom options
```

## 🧠 Pipeline Steps

1. **Data Loading**: Load all rows from activity CSVs, keep only `AccelerationX/Y/Z`, and assign labels from folder names.
2. **Preprocessing**: Handle missing values and standardize using `StandardScaler`.
3. **Model Training**: Train XGBoost (with optional `GridSearchCV`), Random Forest, SVM, and Logistic Regression models.
4. **Evaluation**: Compute Accuracy, Precision, Recall, F1 score, and confusion matrices.
5. **Export**: Save the best model (optimized for recall), scaler, label encoder, and feature list into the `models/` folder.

## 🔗 Integration Guide

You can easily integrate the deployed model into other Python services:

```python
import joblib

# Load artifacts
model = joblib.load("models/fall_detection_model.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Predict from a single accelerometer reading
x, y, z = 0.5, 9.8, 1.2
features = scaler.transform([[x, y, z]])
prediction = model.predict(features)

# Interpret result
class_label = label_encoder.inverse_transform(prediction)[0]
print(f"Prediction: {class_label}")
```

## 🚀 Deployment Options

- **Edge (ESP32)**: Convert to ONNX → TFLite for on-device inference
- **Backend**: Flask/FastAPI endpoint for real-time prediction
- **Mobile**: Flutter app via platform channels
