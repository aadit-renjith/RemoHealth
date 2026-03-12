"""
Preprocessing Module
====================
Clean the raw accelerometer dataset (x, y, z + label).
Scaling is handled separately after feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(
    df: pd.DataFrame,
    label_col: str = "label",
) -> tuple[pd.DataFrame, LabelEncoder, list[str]]:
    """
    Preprocess the raw accelerometer dataset.

    Steps
    -----
    1. Drop rows with missing / corrupted values.
    2. Remove duplicate rows.
    3. Setup label encoder (0=non_fall, 1=fall).
    4. Replace infinities.

    Note: Scaling is NOT done here — it runs after feature engineering
    in main.py so that all features (original + derived) are scaled.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with AccelerationX, AccelerationY, AccelerationZ, label.
    label_col : str
        Name of the target column.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame (not scaled).
    label_encoder : LabelEncoder
        Fitted label encoder.
    feature_cols : list[str]
        List of feature column names used.
    """
    df_clean = df.copy()

    # ── 1. Handle missing values ──
    n_before = len(df_clean)
    df_clean.dropna(inplace=True)
    n_after = len(df_clean)
    if n_before != n_after:
        print(f"   ↳ Dropped {n_before - n_after} rows with missing values")

    # ── 2. Remove duplicates ──
    n_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    n_after = len(df_clean)
    if n_before != n_after:
        print(f"   ↳ Dropped {n_before - n_after} duplicate rows")

    # ── 3. Setup label encoder ──
    label_encoder = LabelEncoder()
    label_encoder.fit(["non_fall", "fall"])  # 0=non_fall, 1=fall
    print(f"   ↳ Labels: 0=Non-Fall, 1=Fall")

    # ── 4. Identify feature columns ──
    feature_cols = [c for c in df_clean.columns if c != label_col]
    print(f"   ↳ Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── 5. Replace infinities with NaN, then fill ──
    df_clean[feature_cols] = df_clean[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    nan_count = df_clean[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        print(f"   ↳ Filling {nan_count} inf/NaN values with column medians")
        df_clean[feature_cols] = df_clean[feature_cols].fillna(
            df_clean[feature_cols].median()
        )

    # Final summary
    print(f"   ↳ Final dataset shape: {df_clean.shape}")
    print(f"   ↳ Class distribution after preprocessing:")
    print(f"      {df_clean[label_col].value_counts().to_dict()}")

    return df_clean, label_encoder, feature_cols
