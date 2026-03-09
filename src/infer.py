from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Load config ───────────────────────────────────────────────────────────────

def load_cfg(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Load saved models ─────────────────────────────────────────────────────────

def load_detector(models_dir: str):
    path = Path(models_dir) / "hybrid_detector.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}\n"
            "Run the full pipeline first: python -m src.main --config config.yaml"
        )
    print(f"  Loaded model from {path}")
    return joblib.load(path)


def load_threshold(models_dir: str) -> float:
    path = Path(models_dir) / "threshold.json"
    if path.exists():
        with open(path) as f:
            return float(json.load(f)["threshold"])
    # Fallback: check metrics.json
    metrics_path = Path(models_dir).parent / "reports" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            t = data.get("threshold")
            if t is not None:
                return float(t)
    print("  WARNING: No saved threshold found, defaulting to 0.1628")
    return 0.1628


# ── Load and prepare new data ─────────────────────────────────────────────────

def load_new_data(csv_path: str, cfg: Dict):
    """
    Loads a new CSV, applies the same normalisation as the training pipeline:
    - Replace inf with NaN
    - Strip and normalise labels
    - Drop configured columns
    """
    print(f"  Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    target_col = cfg["dataset"].get("target_col", " Label")

    # Normalise label column
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str).str.strip().str.upper()
        df[target_col] = (df[target_col] != "BENIGN").astype(int)
    else:
        raise ValueError(
            f"Target column '{target_col}' not found in {csv_path}.\n"
            f"Available columns: {list(df.columns[:10])} ..."
        )

    drop_cols = cfg["dataset"].get("drop_cols", []) or []
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    print(f"  Rows: {len(df):,}  |  Attacks: {y.sum():,}  |  Benign: {(y==0).sum():,}")
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(scores: np.ndarray, y, threshold: float) -> Dict:
    preds = (scores >= threshold).astype(int)
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()

    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = 0.0

    attack = report.get("1", {})
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {report.get('accuracy', 0):.4f}")
    print(f"    Precision (attack): {attack.get('precision', 0):.4f}")
    print(f"    Recall    (attack): {attack.get('recall', 0):.4f}")
    print(f"    F1        (attack): {attack.get('f1-score', 0):.4f}")
    print(f"    Confusion matrix:")
    print(f"      TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"      FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")

    return {"auc": auc, "report": report, "confusion_matrix": cm}


# ── Adversarial helpers ───────────────────────────────────────────────────────

def run_adversarial(detector, X, y, threshold: float, cfg: Dict) -> Dict:
    from .adversarial import (
        add_gaussian_noise, feature_mask,
        fgsm_numeric, pgd_numeric,
        train_numeric_surrogate, train_nonlinear_surrogate,
    )

    adv_cfg  = cfg.get("adversarial", {})
    epsilon  = adv_cfg.get("epsilon", 0.05)
    noise_std= adv_cfg.get("noise_std", 0.02)
    mask_ratio=adv_cfg.get("feature_mask_ratio", 0.05)
    pgd_steps= adv_cfg.get("pgd_steps", 10)
    rs       = cfg.get("dataset", {}).get("random_state", 42)

    # Identify numeric columns (same logic as preprocessing)
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)

    # Use the new data itself as surrogate training data
    # (in a real deployment you'd use the original training data)
    y_series = pd.Series(y)

    print("\n  [Adversarial] Training surrogates on new data ...")
    lin_sur  = train_numeric_surrogate(X, y_series, numeric_cols)
    nl_sur   = train_nonlinear_surrogate(X, y_series, numeric_cols)

    results = {}

    print("\n  [Adversarial] Gaussian noise ...")
    X_noise = add_gaussian_noise(X, numeric_cols, noise_std, random_state=rs)
    results["gaussian_noise"] = compute_metrics(
        detector.predict_proba(X_noise), y, threshold)

    print("\n  [Adversarial] Feature mask ...")
    X_mask  = feature_mask(X, numeric_cols, mask_ratio, random_state=rs)
    results["feature_mask"] = compute_metrics(
        detector.predict_proba(X_mask), y, threshold)

    print("\n  [Adversarial] FGSM ...")
    X_fgsm  = fgsm_numeric(lin_sur, X, epsilon, numeric_cols)
    results["fgsm_numeric"] = compute_metrics(
        detector.predict_proba(X_fgsm), y, threshold)

    print("\n  [Adversarial] PGD (may take a few minutes) ...")
    X_pgd   = pgd_numeric(nl_sur, X, epsilon, numeric_cols, steps=pgd_steps)
    results["pgd_numeric"] = compute_metrics(
        detector.predict_proba(X_pgd), y, threshold)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run IDS inference on a new dataset")
    parser.add_argument("--config",         default="config.yaml")
    parser.add_argument("--data",           required=True, help="Path to new CSV file")
    parser.add_argument("--threshold",      type=float,    default=None,
                        help="Override decision threshold (default: use saved value)")
    parser.add_argument("--no-adversarial", action="store_true",
                        help="Skip adversarial evaluation")
    parser.add_argument("--output",         default=None,
                        help="Output JSON path (default: artifacts/reports/infer_results.json)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    models_dir = cfg.get("paths", {}).get("models_dir", "artifacts/models")

    print("\n" + "="*55)
    print("  Hybrid IDS — Inference on New Dataset")
    print("="*55)

    # Load model
    print("\n[1/4] Loading saved model ...")
    detector = load_detector(models_dir)

    # Load threshold
    threshold = args.threshold if args.threshold is not None else load_threshold(models_dir)
    print(f"  Using threshold: {threshold:.4f}")

    # Load new data
    print("\n[2/4] Loading new dataset ...")
    X, y = load_new_data(args.data, cfg)

    # Clean evaluation
    print("\n[3/4] Clean evaluation ...")
    scores = detector.predict_proba(X)
    clean_results = compute_metrics(scores, y, threshold)

    # Adversarial evaluation
    adv_results = {}
    if not args.no_adversarial:
        print("\n[4/4] Adversarial evaluation ...")
        adv_results = run_adversarial(detector, X, y, threshold, cfg)
    else:
        print("\n[4/4] Skipping adversarial evaluation (--no-adversarial)")

    # Save results
    output_path = args.output or os.path.join(
        cfg.get("paths", {}).get("reports_dir", "artifacts/reports"),
        "infer_results.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "dataset":   args.data,
        "threshold": threshold,
        "clean":     clean_results,
        "adversarial": adv_results,
    }
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Results saved to {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()