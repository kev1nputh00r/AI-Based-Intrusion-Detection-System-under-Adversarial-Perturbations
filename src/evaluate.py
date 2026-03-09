from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from .adversarial import (
    add_gaussian_noise,
    feature_mask,
    fgsm_numeric,
    pgd_numeric,
    train_numeric_surrogate,
    train_nonlinear_surrogate,
)
from .models import HybridDetector


def _compute_metrics(scores: np.ndarray, y, threshold: float, cfg: Dict) -> Dict:
    """Shared metrics computation used by clean and adversarial evaluation."""
    preds = (scores >= threshold).astype(int)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = 0.0

    result: Dict = {"report": report, "auc": auc}

    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("export_confusion_matrix", True):
        result["confusion_matrix"] = confusion_matrix(y, preds).tolist()

    if eval_cfg.get("export_roc_points", True):
        try:
            fpr, tpr, thresholds = roc_curve(y, scores)
            result["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except ValueError:
            result["roc_curve"] = None

    return result


def evaluate_clean(
    detector: HybridDetector,
    X_test,
    y_test,
    threshold: float,
    cfg: Dict,
) -> Dict:
    scores = detector.predict_proba(X_test)
    return _compute_metrics(scores, y_test, threshold, cfg)


def evaluate_adversarial(
    detector: HybridDetector,
    X_train,
    X_test,
    y_test,
    threshold: float,
    numeric_cols: List[str],
    cfg: Dict,
) -> Dict:
    adv_cfg = cfg["adversarial"]
    epsilon = adv_cfg.get("epsilon", 0.05)
    noise_std = adv_cfg.get("noise_std", 0.02)
    mask_ratio = adv_cfg.get("feature_mask_ratio", 0.05)
    pgd_steps = adv_cfg.get("pgd_steps", 10)
    random_state = cfg.get("dataset", {}).get("random_state", 42)

    numeric_cols = list(numeric_cols)
    y_train = cfg["_y_train"]

    print("  Training linear surrogate (FGSM)...")
    linear_surrogate = train_numeric_surrogate(X_train, y_train, numeric_cols)

    print("  Training non-linear surrogate (PGD)...")
    nonlinear_surrogate = train_nonlinear_surrogate(X_train, y_train, numeric_cols)

    print("  Generating adversarial examples...")
    X_noise = add_gaussian_noise(X_test, numeric_cols, noise_std, random_state=random_state)
    X_mask = feature_mask(X_test, numeric_cols, mask_ratio, random_state=random_state)
    X_fgsm = fgsm_numeric(linear_surrogate, X_test, epsilon, numeric_cols)

    print("  Running PGD attack (this may take a moment)...")
    X_pgd = pgd_numeric(nonlinear_surrogate, X_test, epsilon, numeric_cols, steps=pgd_steps)

    return {
        "gaussian_noise": _compute_metrics(
            detector.predict_proba(X_noise), y_test, threshold, cfg
        ),
        "feature_mask": _compute_metrics(
            detector.predict_proba(X_mask), y_test, threshold, cfg
        ),
        "fgsm_numeric": _compute_metrics(
            detector.predict_proba(X_fgsm), y_test, threshold, cfg
        ),
        "pgd_numeric": _compute_metrics(
            detector.predict_proba(X_pgd), y_test, threshold, cfg
        ),
    }