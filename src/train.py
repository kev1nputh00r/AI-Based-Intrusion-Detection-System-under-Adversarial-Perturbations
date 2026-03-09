from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import load_dataset
from .models import HybridDetector
from .preprocessing import build_preprocessor
from .utils import Paths, ensure_dirs


def find_threshold(scores: np.ndarray, y: np.ndarray, target_fpr: float = 0.05) -> float:
    """Select threshold on normal-class scores to bound false positive rate."""
    normal_scores = scores[y == 0]
    if len(normal_scores) == 0:
        return 0.5
    return float(np.quantile(normal_scores, 1 - target_fpr))


def train(cfg: Dict) -> Tuple[HybridDetector, float, Tuple]:
    X_train_full, X_test, y_train_full, y_test = load_dataset(cfg)

    # Hold out a validation split from training data for threshold selection
    random_state = cfg.get("dataset", {}).get("random_state", 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train_full,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train, cfg)

    detector = HybridDetector(preprocessor, cfg)
    detector.fit(X_train, y_train)

    # Threshold selected on held-out validation set (not training data)
    val_scores = detector.predict_proba(X_val)
    target_fpr = cfg.get("evaluation", {}).get("threshold_target_fpr", 0.05)
    threshold = find_threshold(val_scores, y_val, target_fpr=target_fpr)

    return detector, threshold, (X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)


def save_models(detector: HybridDetector, paths: Paths) -> None:
    ensure_dirs(paths)
    # Save full detector
    joblib.dump(detector, f"{paths.models_dir}/hybrid_detector.joblib")
    # Save components separately for flexibility
    if detector.models is not None:
        joblib.dump(detector.models.supervised, f"{paths.models_dir}/supervised_model.joblib")
        joblib.dump(detector.models.anomaly, f"{paths.models_dir}/anomaly_model.joblib")





















