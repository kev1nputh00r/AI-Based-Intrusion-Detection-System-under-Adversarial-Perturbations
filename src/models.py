from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM


@dataclass
class HybridModels:
    supervised: object  # RandomForestClassifier or CalibratedClassifierCV
    anomaly: object     # IsolationForest or OneClassSVM
    hybrid_weight: float


class HybridDetector:
    def __init__(self, preprocessor, cfg: Dict):
        self.preprocessor = preprocessor
        self.cfg = cfg
        self.models: HybridModels | None = None

    def fit(self, X, y) -> None:
        # Build models
        supervised_base = build_supervised_model(self.cfg, calibrate=False)
        anomaly = build_anomaly_model(self.cfg)

        X_processed = self.preprocessor.fit_transform(X)

        # Fit supervised model
        # If calibrating, CalibratedClassifierCV fits via internal cross-val (cv=3)
        if self.cfg["model"].get("calibrate", False):
            supervised = CalibratedClassifierCV(
                supervised_base, method="sigmoid", cv=3
            )
            supervised.fit(X_processed, y)
        else:
            supervised_base.fit(X_processed, y)
            supervised = supervised_base

        # Fit anomaly model on normal samples only
        normal_mask = (y == 0).to_numpy() if hasattr(y, "to_numpy") else (y == 0)
        anomaly.fit(X_processed[normal_mask])

        self.models = HybridModels(
            supervised=supervised,
            anomaly=anomaly,
            hybrid_weight=self.cfg["model"].get("hybrid_weight", 0.7),
        )

    def predict_proba(self, X) -> np.ndarray:
        self._ensure_fit()
        X_processed = self.preprocessor.transform(X)
        supervised_probs = self.models.supervised.predict_proba(X_processed)[:, 1]
        anomaly_scores = -self.models.anomaly.score_samples(X_processed)
        anomaly_scores = self._min_max(anomaly_scores)
        hybrid = (
            self.models.hybrid_weight * supervised_probs
            + (1.0 - self.models.hybrid_weight) * anomaly_scores
        )
        return hybrid

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def evaluate_auc(self, X, y) -> float:
        scores = self.predict_proba(X)
        return roc_auc_score(y, scores)

    def _ensure_fit(self) -> None:
        if self.models is None:
            raise RuntimeError("HybridDetector has not been fitted. Call fit() first.")

    @staticmethod
    def _min_max(values: np.ndarray) -> np.ndarray:
        vmin, vmax = np.min(values), np.max(values)
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)


def build_supervised_model(cfg: Dict, calibrate: bool | None = None):
    """Build supervised model. calibrate=None reads from config."""
    training_cfg = cfg["training"]
    model_name = cfg["model"].get("supervised", "random_forest")
    random_state = training_cfg.get("random_state", 42)
    should_calibrate = calibrate if calibrate is not None else cfg["model"].get("calibrate", False)

    if model_name == "gradient_boosting":
        base_model = GradientBoostingClassifier(
            n_estimators=training_cfg.get("n_estimators", 300),
            max_depth=training_cfg.get("max_depth", 3),
            random_state=random_state,
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=training_cfg.get("n_estimators", 300),
            max_depth=training_cfg.get("max_depth"),
            class_weight=training_cfg.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=-1,
        )

    if should_calibrate:
        return CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    return base_model


def build_anomaly_model(cfg: Dict):
    model_name = cfg["model"].get("anomaly", "isolation_forest")
    random_state = cfg["training"].get("random_state", 42)
    if model_name == "one_class_svm":
        return OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
    )