"""
Model Training Module
=====================
Train and compare multiple classifiers for fall detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data into train and test sets with stratification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"   ↳ Train set: {X_train.shape[0]} samples")
    print(f"   ↳ Test set:  {X_test.shape[0]} samples")
    print(f"   ↳ Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   ↳ Test class distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to handle class imbalance in the training set.

    Returns
    -------
    X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   ↳ After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def _compute_scale_pos_weight(y_train) -> float:
    """Compute scale_pos_weight for XGBoost (ratio of neg/pos samples)."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_xgboost(
    X_train, y_train,
    tune_hyperparameters: bool = True,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with optional hyperparameter tuning.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    tune_hyperparameters : bool
        Whether to perform GridSearchCV tuning.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    XGBClassifier
        Best trained model.
    """
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    print(f"   ↳ scale_pos_weight = {scale_pos_weight:.2f}")

    if tune_hyperparameters:
        print("   ↳ Running hyperparameter tuning (GridSearchCV)...")

        base_model = XGBClassifier(

            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )

        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring="recall",  # Prioritize recall for fall detection
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        print(f"   ↳ Best parameters: {grid_search.best_params_}")
        print(f"   ↳ Best CV recall: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    else:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,

            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_leaf_nodes=64,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model





def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train, y_train,
    tune_xgboost: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Train all four classifiers and return them in a dictionary.

    Returns
    -------
    dict
        {model_name: trained_model}
    """
    models = {}

    print("\n🔧 Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train, tune_xgboost, random_state)

    print("\n🔧 Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train, random_state)



    print("\n🔧 Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train, random_state)

    print("\n✅ All models trained successfully.")
    return models


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """
    Perform stratified cross-validation and return scores.

    Returns
    -------
    dict
        {metric_name: (mean, std)}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=metric, n_jobs=-1)
        metrics[metric] = (scores.mean(), scores.std())

    return metrics
