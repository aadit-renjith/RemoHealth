"""
Feature Engineering Module
===========================
Derive physics-based features from raw x, y, z accelerometer readings
to improve classifier discrimination between falls and non-falls.
"""

import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Add derived features from raw AccelerationX, AccelerationY, AccelerationZ.

    New features (all per-row, no windowing):
    - magnitude: sqrt(x² + y² + z²)
    - horizontal_mag: sqrt(x² + y²)
    - tilt_angle_xy: atan2(y, x)
    - tilt_angle_z: atan2(z, horizontal_mag)
    - x_sq, y_sq, z_sq: squared acceleration terms

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with AccelerationX, Y, Z columns.
    feature_cols : list[str]
        Names of raw feature columns.
    label_col : str
        Name of the target label column.

    Returns
    -------
    pd.DataFrame
        DataFrame with original + new engineered columns.
    """
    df = df.copy()

    x = df["AccelerationX"]
    y = df["AccelerationY"]
    z = df["AccelerationZ"]

    # ── Magnitude features ──
    df["magnitude"] = np.sqrt(x**2 + y**2 + z**2)
    df["horizontal_mag"] = np.sqrt(x**2 + y**2)

    # ── Tilt / orientation angles ──
    df["tilt_angle_xy"] = np.arctan2(y, x)
    df["tilt_angle_z"] = np.arctan2(z, df["horizontal_mag"].replace(0, 1e-10))

    # ── Squared terms (capture non-linear patterns) ──
    df["x_sq"] = x**2
    df["y_sq"] = y**2
    df["z_sq"] = z**2

    new_cols = [
        "magnitude", "horizontal_mag",
        "tilt_angle_xy", "tilt_angle_z",
        "x_sq", "y_sq", "z_sq",
    ]
    print(f"   ↳ Added {len(new_cols)} engineered features: {new_cols}")

    return df


def get_feature_columns(df: pd.DataFrame, label_col: str = "label") -> list[str]:
    """Return list of feature column names (excludes label)."""
    return [c for c in df.columns if c != label_col]
