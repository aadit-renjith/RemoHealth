"""
Fall Detection Model — Main Training Pipeline
==============================================

Orchestrates the full ML pipeline using raw x, y, z accelerometer readings:
  1. Data loading & inspection (folder-based CSVs → raw rows)
  2. Preprocessing (standardization)
  3. Model training (XGBoost, Random Forest, SVM, Logistic Regression)
  4. Evaluation & comparison
  5. Model export

Usage
-----
    python main.py
    python main.py --no-tune
    python main.py --no-tune --no-smote --test-size 0.3
"""

import os
import sys
import argparse
import time

from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset, inspect_dataset
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, get_feature_columns
from src.model_training import split_data, apply_smote, train_all_models, cross_validate_model
from src.evaluation import (
    evaluate_all_models,
    print_classification_report,
    plot_all_confusion_matrices,
    plot_feature_importance,
)
from src.export_model import export_model


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Model Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to the data root directory containing activity subfolders (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Directory to save model outputs (default: models)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip XGBoost hyperparameter tuning (faster)",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Skip SMOTE oversampling",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "=" * 60)
    print("🏥 FALL DETECTION MODEL — TRAINING PIPELINE")
    print("   (Raw x, y, z accelerometer features)")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # STEP 1: Load & Inspect Data
    # ──────────────────────────────────────────────
    print("\n📂 STEP 1: Loading Dataset")
    print("-" * 40)
    print(f"   Data directory: {os.path.abspath(args.data)}")
    df = load_dataset(args.data)
    summary = inspect_dataset(df)
    label_col = summary["label_column"]

    # ──────────────────────────────────────────────
    # STEP 2: Preprocessing
    # ──────────────────────────────────────────────
    print("\n🔧 STEP 2: Preprocessing")
    print("-" * 40)
    df_clean, label_encoder, feature_cols = preprocess(df, label_col)

    # ──────────────────────────────────────────────
    # STEP 3: Feature Engineering (pass-through)
    # ──────────────────────────────────────────────
    print("\n⚙️  STEP 3: Feature Engineering")
    print("-" * 40)
    df_featured = engineer_features(df_clean, feature_cols, label_col)
    feature_cols = get_feature_columns(df_featured, label_col)

    # ── Scale ALL features (original + engineered) ──
    print("\n📐 STEP 3b: Scaling Features")
    print("-" * 40)
    scaler = StandardScaler()
    df_featured[feature_cols] = scaler.fit_transform(df_featured[feature_cols].values)
    print(f"   ↳ Scaled {len(feature_cols)} features (mean=0, std=1)")

    # ──────────────────────────────────────────────
    # STEP 4: Train/Test Split
    # ──────────────────────────────────────────────
    print("\n📊 STEP 4: Train/Test Split")
    print("-" * 40)
    X_train, X_test, y_train, y_test = split_data(
        df_featured, feature_cols, label_col, test_size=args.test_size
    )

    # Optional SMOTE
    if not args.no_smote:
        print("\n⚖️  Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train)

    # ──────────────────────────────────────────────
    # STEP 5: Model Training
    # ──────────────────────────────────────────────
    print("\n🚀 STEP 5: Training Models")
    print("-" * 40)
    models = train_all_models(
        X_train, y_train,
        tune_xgboost=(not args.no_tune),
    )

    # ──────────────────────────────────────────────
    # STEP 6: Evaluation
    # ──────────────────────────────────────────────
    print("\n📈 STEP 6: Model Evaluation")
    print("-" * 40)

    # Compare all models
    df_results = evaluate_all_models(models, X_test, y_test)

    # Detailed report for best model
    best_row = df_results.iloc[0]
    best_model_name = best_row["model"]
    best_model = models[best_model_name]

    print_classification_report(best_model, X_test, y_test, best_model_name)



    # Confusion matrices for all models
    print("\n📊 Generating confusion matrices...")
    plot_all_confusion_matrices(models, X_test, y_test, args.output)

    # Feature importance for tree-based models
    print("\n📊 Generating feature importance plots...")
    for name in ["XGBoost", "Random Forest"]:
        if name in models:
            plot_feature_importance(models[name], feature_cols, name, args.output)

    # ──────────────────────────────────────────────
    # STEP 7: Export Best Model
    # ──────────────────────────────────────────────
    print("\n💾 STEP 7: Exporting Model")
    print("-" * 40)

    print(f"   Best model by recall: {best_model_name}")
    print(f"   Recall: {best_row['recall']:.4f} | F1: {best_row['f1_score']:.4f} | Acc: {best_row['accuracy']:.4f}")

    export_model(
        model=best_model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_list=feature_cols,
        save_dir=args.output,
    )

    # ──────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Best model: {best_model_name}")
    print(f"   Recall (Fall): {best_row['recall']:.4f}")
    print(f"   F1-Score:      {best_row['f1_score']:.4f}")
    print(f"   Accuracy:      {best_row['accuracy']:.4f}")
    print(f"   Features:      {feature_cols}")
    print(f"   Exported to:   {args.output}/")
    print()
    print("\n📌 Deployment — predict from a single (x, y, z) reading:")
    print("   See feature_engineering.py for the required feature derivations.")
    print("   model = joblib.load('models/fall_detection_model.joblib')")
    print("   scaler = joblib.load('models/scaler.joblib')")
    print("   # 1 = Fall, 0 = Non-Fall")
    print("=" * 60)


if __name__ == "__main__":
    main()
