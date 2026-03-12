# Health Anomaly Detection Model

This repository contains the machine learning pipeline for detecting health anomalies using wearable sensor data (e.g., from an ESP32 microcontroller). The goal is to provide a reliable, real-time safety monitoring system.

## Overview

The core of this project is a binary classification model built using XGBoost. It analyzes physiological readings to determine if a patient's current state is **Normal (0)** or **Abnormal (1)**.

Because this is a critical healthcare application, the model is optimized specifically for **High Recall** on the "Abnormal" class. This tuning minimizes false negatives, ensuring that dangerous health anomalies are rarely missed by the system.

## Project Structure

```text
Health_Anomaly_Detection_Model/
├── data/                       # Contains necessary datasets (ignored by Git)
├── train_xgboost_model.py      # The main training pipeline script
├── xgboost_health_model.pkl    # The serialized, trained XGBoost model
├── model_feature_info.txt      # Information detailing the expected features and their input order
├── confusion_matrix.png        # Generated plot: performance breakdown of predictions
├── feature_importance.png      # Generated plot: importance of each feature in the tree
└── roc_curve.png               # Generated plot: Receiver Operating Characteristic curve
```

## Features

The model processes the following feature engineering points from raw data:
- Splits standard Blood Pressure readings (e.g., 120/80) into distinct **Systolic** and **Diastolic** numerical features.
- Computes the **Mean Arterial Pressure (MAP)** using the formula: `(Systolic + 2 * Diastolic) / 3`.
- Handles missing values by prioritizing dataset integrity over aggressive imputation.
- Performs target class weighting (`scale_pos_weight`) automatically during training to handle dataset imbalances.

Note: Because XGBoost is a tree-based model, standard dataset feature scaling (like StandardScaler or MinMaxScaler) is completely bypassed. This simplifies deployment pipelines by allowing raw sensor data to be passed directly to the model.

## Setup and Training

### Prerequisites

Ensure you have the required Python libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### Running the Pipeline

To re-run the training pipeline and generate a new model:

1. Ensure your dataset is present as specified in the script (`healthcare_training_ready_dataset.csv` or similar depending on how `train_xgboost_model.py` is configured). Look at the `DATASET_PATH` variable inside the script.
2. Run the script:

   ```bash
   python train_xgboost_model.py
   ```

3. The script will automatically output evaluation metrics (Accuracy, Precision, Recall, F1), generate visualization plots, and save the updated `.pkl` model and feature configuration files in the root folder.
