from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from .models import HybridDetector


def run_ablation(
    detector: HybridDetector,
    X_test,
    y_test,
    thresholds: List[float],
) -> Dict:
    scores = detector.predict_proba(X_test)

    # AUC is threshold-independent — compute once outside the loop
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.0

    results: Dict[str, Dict] = {}
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        results[str(threshold)] = {
            "threshold": float(threshold),
            "auc": auc,
            "report": report,
        }

    return results




















