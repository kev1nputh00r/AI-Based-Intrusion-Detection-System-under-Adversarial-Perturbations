from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ensure_float(X_adv: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Coerce numeric columns to float64 in-place."""
    for col in numeric_cols:
        X_adv[col] = pd.to_numeric(X_adv[col], errors="coerce").astype("float64")


def add_gaussian_noise(
    X: pd.DataFrame,
    numeric_cols: List[str],
    noise_std: float,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Add zero-mean Gaussian noise to numeric features."""
    X_adv = X.copy()
    if not numeric_cols:
        return X_adv
    _ensure_float(X_adv, numeric_cols)
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_std, size=(len(X_adv), len(numeric_cols)))
    X_adv.loc[:, numeric_cols] = X_adv[numeric_cols].values + noise
    return X_adv


def feature_mask(
    X: pd.DataFrame,
    numeric_cols: List[str],
    mask_ratio: float,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Randomly zero out a fraction of numeric features."""
    X_adv = X.copy()
    if not numeric_cols or mask_ratio <= 0:
        return X_adv
    _ensure_float(X_adv, numeric_cols)
    rng = np.random.default_rng(random_state)
    mask = rng.random(size=X_adv[numeric_cols].shape) < mask_ratio
    X_adv.loc[:, numeric_cols] = X_adv[numeric_cols].where(~mask, other=0.0)
    return X_adv


def train_numeric_surrogate(
    X: pd.DataFrame,
    y,
    numeric_cols: List[str],
) -> Pipeline:
    """
    Train a linear SGD surrogate on TRAINING data only.
    Used for FGSM global gradient approximation.
    """
    surrogate = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    scaler = StandardScaler(with_mean=False)
    imputer = SimpleImputer(strategy="median")
    pipeline = Pipeline([("imputer", imputer), ("scaler", scaler), ("model", surrogate)])
    pipeline.fit(X[numeric_cols], y)
    return pipeline


def train_nonlinear_surrogate(
    X: pd.DataFrame,
    y,
    numeric_cols: List[str],
) -> Tuple:
    """
    Train a non-linear GradientBoosting surrogate on TRAINING data only.
    Used for stronger per-sample gradient approximation in PGD.
    Returns (imputer, scaler, surrogate).
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=False)
    surrogate = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )
    X_imp = imputer.fit_transform(X[numeric_cols])
    X_scaled = scaler.fit_transform(X_imp)
    surrogate.fit(X_scaled, y)
    return imputer, scaler, surrogate


def fgsm_numeric(
    surrogate_pipeline: Pipeline,
    X: pd.DataFrame,
    epsilon: float,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Single-step FGSM using sign of linear surrogate coefficients."""
    X_adv = X.copy()
    if not numeric_cols:
        return X_adv
    _ensure_float(X_adv, numeric_cols)

    scaler = surrogate_pipeline.named_steps["scaler"]
    imputer = surrogate_pipeline.named_steps["imputer"]
    model = surrogate_pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        return X_adv

    X_imp = imputer.transform(X_adv[numeric_cols])
    X_proc = scaler.transform(X_imp)
    grad_sign = np.sign(model.coef_[0])
    X_proc_adv = X_proc + epsilon * grad_sign
    X_adv.loc[:, numeric_cols] = scaler.inverse_transform(X_proc_adv)
    return X_adv


def pgd_numeric(
    surrogate_tuple: Tuple,
    X: pd.DataFrame,
    epsilon: float,
    numeric_cols: List[str],
    steps: int = 10,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    """
    PGD (Projected Gradient Descent) attack using a non-linear GradientBoosting surrogate.

    Applies FGSM iteratively using finite-difference gradient approximation.
    Much stronger than single-step FGSM because:
    - Uses non-linear GradientBoosting surrogate for better gradient estimates
    - Iterates multiple steps, accumulating perturbation progressively
    - Per-sample gradients computed at each step
    - Projects back into epsilon ball after each step (L-inf constraint)
    """
    X_adv = X.copy()
    if not numeric_cols:
        return X_adv
    _ensure_float(X_adv, numeric_cols)

    if alpha is None:
        alpha = epsilon / steps

    imputer, scaler, surrogate = surrogate_tuple
    X_imp = imputer.transform(X_adv[numeric_cols])
    X_curr = scaler.transform(X_imp)
    X_orig = X_curr.copy()

    delta = 1e-3  # finite difference step size

    for _ in range(steps):
        proba = surrogate.predict_proba(X_curr)[:, 1]  # shape: (n_samples,)

        # Finite-difference per-sample gradient sign across all features
        grad_signs = np.zeros_like(X_curr)
        for i in range(X_curr.shape[1]):
            X_plus = X_curr.copy()
            X_plus[:, i] += delta
            proba_plus = surrogate.predict_proba(X_plus)[:, 1]
            grad_signs[:, i] = np.sign(proba_plus - proba)

        # Step in gradient direction
        X_curr = X_curr + alpha * grad_signs

        # Project back into L-inf epsilon ball around original
        perturbation = np.clip(X_curr - X_orig, -epsilon, epsilon)
        X_curr = X_orig + perturbation

    X_adv.loc[:, numeric_cols] = scaler.inverse_transform(X_curr)
    return X_adv




























