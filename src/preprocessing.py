from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame, cfg: Dict) -> Tuple[ColumnTransformer, List[str], List[str]]:
    dataset_cfg = cfg["dataset"]
    categorical_cols = dataset_cfg.get("categorical_cols") or []
    if not categorical_cols:
        categorical_cols = list(
            X.select_dtypes(include=["object", "category", "string"]).columns
        )

    numeric_cols = [c for c in X.columns if c not in categorical_cols and is_numeric_dtype(X[c])]

    transformers = []
    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if dataset_cfg.get("normalize", True):
            num_steps.append(("scaler", StandardScaler(with_mean=False)))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols
