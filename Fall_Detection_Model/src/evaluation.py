"""
Evaluation Module
=================
Evaluate trained models and generate comparison reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a single model on the test set.

    Returns
    -------
    dict
        Dictionary of metric values.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.

    Parameters
    ----------
    models : dict
        {model_name: trained_model}
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by recall (descending).
    """
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("recall", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by recall)")
    print("=" * 70)
    print(df_results.to_string(index=False, float_format="%.4f"))
    print("=" * 70)

    return df_results


def print_classification_report(model, X_test, y_test, model_name: str = "Model"):
    """Print the full sklearn classification report."""
    y_pred = model.predict(X_test)
    print(f"\n📊 Classification Report — {model_name}")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=["Non-Fall", "Fall"]))


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    save_dir: str = "models",
) -> str:
    """
    Generate and save a confusion matrix heatmap.

    Returns
    -------
    str
        Path to the saved image.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fall", "Fall"],
        yticklabels=["Non-Fall", "Fall"],
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"   ↳ Confusion matrix saved to: {save_path}")
    return save_path


def plot_all_confusion_matrices(models: dict, X_test, y_test, save_dir: str = "models"):
    """Plot confusion matrices for all models."""
    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, name, save_dir)


def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "XGBoost",
    save_dir: str = "models",
    top_n: int = 20,
) -> str:
    """
    Plot top-N feature importances from a tree-based model.

    Returns
    -------
    str
        Path to the saved image.
    """
    os.makedirs(save_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="steelblue",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()

    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"   ↳ Feature importance plot saved to: {save_path}")
    return save_path
