"""
Data Loader Module
==================
Load accelerometer CSV files from activity-labelled subdirectories
and return a flat DataFrame with raw x, y, z readings + label.

Expected directory layout
-------------------------
    data/
    ├── freeFall/   ← fall
    ├── runFall/    ← fall
    ├── walkFall/   ← fall
    ├── downSit/    ← non-fall
    ├── runSit/     ← non-fall
    └── walkSit/    ← non-fall

Each CSV has semicolon-separated columns:
    Date;Timestamp;DeviceOrientation;AccelerationX;AccelerationY;AccelerationZ;Label
"""

import os
import numpy as np
import pandas as pd


# ── Label mapping ────────────────────────────────────────────
FALL_FOLDERS = {"freefall", "runfall", "walkfall"}


def _is_fall_folder(name: str) -> bool:
    """Check if a folder name corresponds to a fall activity."""
    return name.lower() in FALL_FOLDERS


# ── Main loading functions ───────────────────────────────────
def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV recordings from activity subdirectories and return
    a flat DataFrame with columns: AccelerationX, AccelerationY,
    AccelerationZ, label.

    Each row in each CSV becomes a separate sample.

    Parameters
    ----------
    data_dir : str
        Path to the data root (e.g. ``data/``).

    Returns
    -------
    pd.DataFrame
        One row per accelerometer reading, with columns
        [AccelerationX, AccelerationY, AccelerationZ, label].
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: '{data_dir}'. "
            "Please download the dataset from Kaggle and extract it.\n"
            "URL: https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data/data"
        )

    frames = []
    skipped = 0

    subdirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not subdirs:
        raise FileNotFoundError(
            f"No activity subdirectories found in '{data_dir}'. "
            "Expected folders like freeFall/, walkSit/, etc."
        )

    print(f"   ↳ Found {len(subdirs)} activity folders: {subdirs}")

    for folder in subdirs:
        folder_path = os.path.join(data_dir, folder)
        is_fall = _is_fall_folder(folder)
        label = 1 if is_fall else 0
        label_str = "Fall" if is_fall else "Non-Fall"

        csv_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith(".csv")
        )

        folder_rows = 0
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(csv_path, sep=";")

                # Verify required columns exist
                required = {"AccelerationX", "AccelerationY", "AccelerationZ"}
                if not required.issubset(set(df.columns)):
                    print(f"   ⚠️  Skipping {csv_file}: missing columns {required - set(df.columns)}")
                    skipped += 1
                    continue

                # Keep only the acceleration columns
                df_raw = df[["AccelerationX", "AccelerationY", "AccelerationZ"]].copy()

                # Convert to numeric (coerce errors to NaN)
                for col in df_raw.columns:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

                # Drop rows with NaN
                df_raw = df_raw.dropna()

                if len(df_raw) == 0:
                    skipped += 1
                    continue

                df_raw["label"] = label
                frames.append(df_raw)
                folder_rows += len(df_raw)

            except Exception as e:
                print(f"   ⚠️  Error reading {csv_file}: {e}")
                skipped += 1

        print(f"   ↳ {folder:12s} → {label_str:8s} | {len(csv_files):3d} files | {folder_rows:6d} rows")

    if not frames:
        raise ValueError("No valid recordings loaded. Check data directory structure.")

    df_all = pd.concat(frames, ignore_index=True)

    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} files due to errors")

    print(f"   ↳ Total rows loaded: {len(df_all)}")
    return df_all


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Inspect the dataset and print a summary.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset with AccelerationX/Y/Z + label.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    n_samples, n_cols = df.shape
    print(f"\n📊 Shape: {n_samples} samples × {n_cols} columns")

    # Feature columns
    feature_cols = [c for c in df.columns if c != "label"]
    print(f"\n📋 Feature columns ({len(feature_cols)}): {feature_cols}")

    # First few rows
    print(f"\n📋 First 5 rows:")
    print(df.head().to_string())

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n🔍 Missing Values (total: {total_missing}):")
    print(missing[missing > 0] if total_missing > 0 else "   None found ✅")

    # Class distribution
    class_dist = df["label"].value_counts().sort_index()
    class_map = {0: "Non-Fall", 1: "Fall"}
    print(f"\n🏷️  Class Distribution:")
    for cls, count in class_dist.items():
        print(f"   {class_map.get(cls, cls)}: {count}")

    balance_ratio = class_dist.min() / class_dist.max()
    print(f"   Balance Ratio: {balance_ratio:.3f}")

    summary = {
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "total_missing": total_missing,
        "label_column": "label",
        "class_distribution": class_dist,
    }

    print("=" * 60)
    return summary
