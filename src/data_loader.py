from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _normalize_label_series(labels: pd.Series) -> pd.Series:
    if labels.dtype == object:
        return labels.str.strip().map(
            lambda x: 0 if x.upper() in {"BENIGN", "NORMAL", "0", "FALSE"} else 1
        )
    return labels.astype(int)


def load_dataset(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset_cfg = cfg["dataset"]
    train_csv = dataset_cfg["train_csv"]
    test_csv = dataset_cfg.get("test_csv")

    df_train = pd.read_csv(train_csv)
    if test_csv and Path(test_csv).exists():
        df_test = pd.read_csv(test_csv)
        X_train, y_train = _split_xy(df_train, dataset_cfg)
        X_test, y_test = _split_xy(df_test, dataset_cfg)
        return X_train, X_test, y_train, y_test

    X, y = _split_xy(df_train, dataset_cfg)
    return train_test_split(
        X,
        y,
        test_size=dataset_cfg.get("test_size", 0.2),
        random_state=dataset_cfg.get("random_state", 42),
        stratify=y,
    )


def _split_xy(df: pd.DataFrame, dataset_cfg: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = dataset_cfg["target_col"]
    drop_cols = dataset_cfg.get("drop_cols", [])
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan)

    y = _normalize_label_series(df[target_col])
    X = df.drop(columns=[target_col])
    return X, y




















