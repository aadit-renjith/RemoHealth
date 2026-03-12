"""
RemoHealth Project — XGBoost Classification Model Training
===========================================================
Trains a binary classifier to detect abnormal physiological conditions
from wearable IoT sensor data (ESP32 + sensors).

Classes:
    Normal   → 0
    Abnormal → 1

Optimised for HIGH RECALL on the Abnormal class to minimise
dangerous false negatives in a real-time medical monitoring pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "healthcare_training_ready_dataset.csv")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "xgboost_health_model.pkl")
OUTPUT_DIR = os.path.dirname(__file__)
RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 70)
print("  RemoHealth — XGBoost Model Training Pipeline")
print("=" * 70)

# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────
print("\n📂 Step 1: Loading dataset …")
df = pd.read_csv(DATASET_PATH)
print(f"   Shape : {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 2. MISSING VALUES CHECK
# ──────────────────────────────────────────────────────────────
print("\n🔍 Step 2: Checking for missing values …")
missing = df.isnull().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("   ✅ No missing values found.")
else:
    print(f"   ⚠️  Found {total_missing} missing values:")
    print(missing[missing > 0].to_string())
    # Drop rows with missing values (acceptable for small counts)
    df.dropna(inplace=True)
    print(f"   After dropping: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
print("\n🔧 Step 3: Feature engineering …")

# 3a. Split Blood Pressure into Systolic & Diastolic
bp_col = "Blood Pressure (mmHg)"
bp_split = df[bp_col].astype(str).str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
print(f"   ✅ Split '{bp_col}' → Systolic_BP, Diastolic_BP")

# 3b. Compute Mean Arterial Pressure (MAP)
df["MAP"] = ((df["Systolic_BP"] + 2 * df["Diastolic_BP"]) / 3).round(2)
print("   ✅ Computed Mean Arterial Pressure (MAP)")

# 3c. Drop Timestamp and original Blood Pressure columns
drop_cols = ["Timestamp", bp_col]
df.drop(columns=drop_cols, inplace=True, errors="ignore")
print(f"   ✅ Dropped columns: {drop_cols}")

# 3d. Handle any NaN introduced by coercion
if df.isnull().sum().sum() > 0:
    before = len(df)
    df.dropna(inplace=True)
    print(f"   ⚠️  Dropped {before - len(df)} rows with NaN from BP parsing")

print(f"\n   Final features: {list(df.columns)}")
print(f"   Final shape  : {df.shape}")

# ──────────────────────────────────────────────────────────────
# 4. LABEL ENCODING
# ──────────────────────────────────────────────────────────────
print("\n🏷️  Step 4: Encoding target labels …")
label_map = {"Normal": 0, "Abnormal": 1}
df["Label"] = df["Label"].map(label_map)

label_counts = df["Label"].value_counts()
print(f"   Normal (0)   : {label_counts.get(0, 0)}")
print(f"   Abnormal (1) : {label_counts.get(1, 0)}")

n_normal = label_counts.get(0, 1)
n_abnormal = label_counts.get(1, 1)
scale_pos_weight = n_normal / n_abnormal
print(f"   scale_pos_weight (imbalance ratio): {scale_pos_weight:.2f}")

# ──────────────────────────────────────────────────────────────
# 5. VERIFY ALL FEATURES ARE NUMERIC
# ──────────────────────────────────────────────────────────────
print("\n🔢 Step 5: Verifying all features are numeric …")
print(f"   Dtypes:\n{df.dtypes.to_string()}")
assert df.select_dtypes(include=[np.number]).shape[1] == df.shape[1], \
    "Non-numeric columns still present!"
print("   ✅ All columns are numeric.")

# ──────────────────────────────────────────────────────────────
# 6. SEPARATE FEATURES & TARGET
# ──────────────────────────────────────────────────────────────
X = df.drop(columns=["Label"])
y = df["Label"]
print(f"\n   Features (X): {X.shape}")
print(f"   Target   (y): {y.shape}")
print(f"   Feature names: {list(X.columns)}")

# ──────────────────────────────────────────────────────────────
# 7. TRAIN-TEST SPLIT
# ──────────────────────────────────────────────────────────────
print("\n✂️  Step 6: Train-Test Split (80/20, stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test  set: {X_test.shape[0]} samples")
print(f"   Train label distribution:\n{y_train.value_counts().to_string()}")
print(f"   Test  label distribution:\n{y_test.value_counts().to_string()}")

# ──────────────────────────────────────────────────────────────
# 8. SCALING DECISION
# ──────────────────────────────────────────────────────────────
print("\n📐 Step 7: Scaling decision …")
print("   ℹ️  XGBoost is a tree-based ensemble model that uses split-based")
print("      decisions, NOT distance-based computations. Therefore, standard")
print("      scaling / normalisation is NOT required and is SKIPPED.")
print("   ✅ This also benefits deployment: raw sensor values can be fed directly.")

# ──────────────────────────────────────────────────────────────
# 9. HYPERPARAMETER TUNING (RandomizedSearchCV)
# ──────────────────────────────────────────────────────────────
print("\n⚡ Step 8: Hyperparameter tuning with RandomizedSearchCV …")
print("   Scoring metric: recall (to minimise false negatives)")

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=60,
    scoring="recall",
    cv=cv_strategy,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    return_train_score=True,
)

random_search.fit(X_train, y_train)

print("\n   ✅ Tuning complete!")
print(f"   Best CV Recall: {random_search.best_score_:.4f}")
print(f"   Best Parameters:")
for param, val in random_search.best_params_.items():
    print(f"      {param}: {val}")

best_model = random_search.best_estimator_

# ──────────────────────────────────────────────────────────────
# 10. EVALUATION ON TEST SET
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  📊 EVALUATION ON TEST SET")
print("=" * 70)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n   Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}  ⬅️  CRITICAL for medical safety")
print(f"   F1-Score  : {f1:.4f}")
print(f"   ROC-AUC   : {roc_auc:.4f}")

print("\n   📋 Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Normal (0)", "Abnormal (1)"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("   🔢 Confusion Matrix:")
print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
print(f"      FN={cm[1][0]}  TP={cm[1][1]}")

if cm.shape[0] > 1:
    fn = cm[1][0]
    tp = cm[1][1]
    if fn == 0:
        print("\n   🎯 PERFECT — Zero false negatives for Abnormal class!")
    else:
        print(f"\n   ⚠️  {fn} false negative(s) detected — review threshold if needed.")

# ──────────────────────────────────────────────────────────────
# 11. VISUALISATIONS
# ──────────────────────────────────────────────────────────────
print("\n📈 Step 9: Generating visualisations …")

# 11a. Confusion Matrix heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"], ax=ax,
            annot_kws={"size": 16})
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title("Confusion Matrix — XGBoost Health Monitor", fontsize=14, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"   ✅ Saved: {cm_path}")

# 11b. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve — XGBoost Health Monitor", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"   ✅ Saved: {roc_path}")

# 11c. Feature Importance
importances = best_model.feature_importances_
feat_names = X.columns
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(feat_names[sorted_idx], fontsize=11)
ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
ax.set_title("Feature Importance — XGBoost Health Monitor", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"   ✅ Saved: {fi_path}")

# ──────────────────────────────────────────────────────────────
# 12. SAVE MODEL
# ──────────────────────────────────────────────────────────────
print("\n💾 Step 10: Saving trained model …")
joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"   ✅ Model saved to: {MODEL_SAVE_PATH}")

# Also export feature names for deployment reference
feature_info_path = os.path.join(OUTPUT_DIR, "model_feature_info.txt")
with open(feature_info_path, "w") as f:
    f.write("Feature Order for Model Input\n")
    f.write("=" * 40 + "\n")
    for i, name in enumerate(X.columns):
        f.write(f"{i}: {name}\n")
    f.write(f"\nTotal features: {len(X.columns)}\n")
    f.write(f"Label mapping: Normal=0, Abnormal=1\n")
    f.write(f"\nBest Hyperparameters:\n")
    for param, val in random_search.best_params_.items():
        f.write(f"  {param}: {val}\n")
print(f"   ✅ Feature info saved to: {feature_info_path}")

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ✅ TRAINING PIPELINE COMPLETE")
print("=" * 70)
print(f"   Model          : {MODEL_SAVE_PATH}")
print(f"   Accuracy        : {acc*100:.2f}%")
print(f"   Recall (Abnormal): {rec*100:.2f}%")
print(f"   ROC-AUC         : {roc_auc:.4f}")
print(f"   Plots           : confusion_matrix.png, roc_curve.png, feature_importance.png")
print("=" * 70)
