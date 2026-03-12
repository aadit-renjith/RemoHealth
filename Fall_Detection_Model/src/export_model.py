"""
Model Export Module
===================
Save the trained model, scaler, label encoder, and feature list for deployment.
"""

import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


def export_model(
    model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    feature_list: list[str],
    save_dir: str = "models",
    model_filename: str = "fall_detection_model.joblib",
):
    """
    Export the full model pipeline for deployment.

    Saves
    -----
    - fall_detection_model.joblib  — trained XGBoost model
    - scaler.joblib                — fitted StandardScaler
    - label_encoder.joblib         — fitted LabelEncoder
    - feature_list.json            — ordered list of feature names
    - preprocessing_pipeline.joblib — bundled (scaler + label_encoder)

    Parameters
    ----------
    model : estimator
        Trained model.
    scaler : StandardScaler
        Fitted scaler.
    label_encoder : LabelEncoder
        Fitted label encoder.
    feature_list : list[str]
        Ordered feature column names.
    save_dir : str
        Output directory.
    model_filename : str
        Filename for the model file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler saved to: {scaler_path}")

    # Save label encoder
    le_path = os.path.join(save_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)
    print(f"   ✅ Label encoder saved to: {le_path}")

    # Save feature list
    feature_path = os.path.join(save_dir, "feature_list.json")
    with open(feature_path, "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"   ✅ Feature list saved to: {feature_path}")

    # Save bundled preprocessing pipeline
    pipeline = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_list": feature_list,
    }
    pipeline_path = os.path.join(save_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"   ✅ Preprocessing pipeline saved to: {pipeline_path}")

    print(f"\n🎉 All artifacts exported to '{save_dir}/' directory.")
    print("   Files:")
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"   • {fname} ({size_kb:.1f} KB)")
