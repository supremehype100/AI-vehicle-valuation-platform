from __future__ import annotations

"""
training/metrics.py

Regression evaluation metrics shared by every training pipeline (private_csv,
private_traintest, dealer_traintest, salvage_traintest) and by
training/evaluate.py's comparison report. Pure numpy — no DB, no I/O.
"""

import numpy as np

# Percentiles reported inside the "error_distribution" sub-dict.
_ERROR_DIST_PERCENTILES = (10, 25, 50, 75, 90)

# Dollar thresholds for the pct_within_$X metrics.
_WITHIN_THRESHOLDS = {
    500: "pct_within_500",
    1000: "pct_within_1000",
    2500: "pct_within_2500",
    5000: "pct_within_5000",
}


def compute_metrics(y_true, y_pred, mape_epsilon: float = 1e-6) -> dict:
    """
    Compute a standard bundle of regression metrics comparing predicted vs.
    actual prices.

    Parameters
    ----------
    y_true, y_pred : array-like of the same length
    mape_epsilon : float
        MAPE guard. Rows where abs(y_true) <= mape_epsilon are excluded from
        the MAPE calculation (a division by ~zero) rather than raising
        ZeroDivisionError or polluting the average with a near-infinite
        term. In this project prices are always >= $1000 by construction
        (see Features.is_valid_price), so in practice no rows should ever be
        excluded — the guard exists because this function is generic and
        must not crash or silently return garbage if it's ever handed data
        that doesn't satisfy that invariant (e.g. a unit test, or a future
        caller with a different price floor). If every row is excluded,
        `mape` is returned as NaN rather than raising.

    Returns
    -------
    dict with keys:
      mae, rmse, r2, median_abs_error, mape, bias,
      pct_within_500, pct_within_1000, pct_within_2500, pct_within_5000,
      error_distribution (dict of p10/p25/p50/p75/p90 of ABSOLUTE error),
      n (row count actually scored)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("compute_metrics called with empty arrays")

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_abs_error = float(np.median(abs_errors))
    bias = float(np.mean(errors))  # mean(pred - true): positive = model overshoots on average

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    valid_mask = np.abs(y_true) > mape_epsilon
    if valid_mask.any():
        mape = float(np.mean(abs_errors[valid_mask] / np.abs(y_true[valid_mask])) * 100.0)
    else:
        mape = float("nan")

    pct_within = {
        name: float(np.mean(abs_errors <= threshold))
        for threshold, name in _WITHIN_THRESHOLDS.items()
    }

    error_distribution = {
        f"p{p}": float(np.percentile(abs_errors, p)) for p in _ERROR_DIST_PERCENTILES
    }

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_abs_error": median_abs_error,
        "mape": mape,
        "bias": bias,
        **pct_within,
        "error_distribution": error_distribution,
        "n": int(y_true.size),
    }
