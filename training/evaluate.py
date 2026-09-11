from __future__ import annotations

"""
training/evaluate.py

Cross-pipeline comparison and evaluation reporting. Generic across all four
model types (private_csv, private_traintest, dealer_traintest,
salvage_traintest) because every one of them pickles an
(XGBRegressor, {col: LabelEncoder}) pair in the same shape.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from training.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_saved_model(artifact_path, encoder_path, X, y_true) -> dict:
    """
    Load a pickled (model, encoders) pair and score it against an
    already-encoded feature matrix X (i.e. the output of
    Features.prepare_features() / prepare_salvage_features()) and the
    corresponding y_true.

    `encoder_path` is loaded too (not just `artifact_path`) even though X is
    expected to already be encoded — this keeps the function's signature
    symmetric with how every pipeline saves its artifacts (model + encoders
    as a pair) and fails loudly/early if the encoder pickle is missing or
    corrupt, rather than silently scoring against a possibly-mismatched X.

    Returns the same dict shape as training.metrics.compute_metrics().
    """
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        pickle.load(f)  # loaded for parity/fail-fast only; X must already be encoded with these

    y_pred = model.predict(X)
    return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Fair private-model comparison — Model A (private_csv) vs Model B
# (private_traintest) scored on the SAME held-out rows.
#
# `score_saved_model` above answers "how does one model do on data I've
# already prepared"; `compare_models`-style reporting that just tabulates
# each model's own self-reported metrics compares Model A on its personal
# CSV against Model B on its own time-split test set — different
# distributions, not the "comparable validation dataset" the task requires.
# This builds ONE held-out set (private_traintest's own time-aware split,
# recomputed fresh against current data) and scores both models against it,
# each with its own fitted encoders, so the comparison is on equal footing.
# ---------------------------------------------------------------------------

def build_private_holdout(session, test_fraction: float = 0.15) -> pd.DataFrame:
    """
    Recomputes the same held-out private test set training.private_traintest
    would carve out right now (dedupe by fingerprint -> time-aware split ->
    VIN-disjoint enforcement -> leakage neutralization), for use as a shared
    evaluation set. Deterministic given the current DB contents, so calling
    this twice in a row (with no scraping in between) returns the same rows.
    """
    from training._traintest_common import _query_listings
    from training import splits, leakage
    from Features import listings_to_dataframe
    from config import LISTING_TYPE_PRIVATE

    listings = _query_listings(session, LISTING_TYPE_PRIVATE)
    listings = splits.dedupe_by_fingerprint(listings)
    train_listings, test_listings = splits.time_aware_split(listings, test_fraction=test_fraction)
    train_listings, test_listings = splits.enforce_vin_disjoint(train_listings, test_listings)
    splits.assert_no_vin_leakage(train_listings, test_listings)
    return leakage.zero_out_leaky_features(listings_to_dataframe(test_listings))


def compare_private_baseline_vs_traintest(
    session,
    test_fraction: float = 0.15,
    output_path: str = "reports/private_model_fair_comparison.md",
) -> list[dict]:
    """
    Scores the latest registered private_csv (Model A) and private_traintest
    (Model B) artifacts against the SAME held-out rows and writes a
    comparison report. Returns [{"model_name": ..., **metrics}, ...].

    KNOWN ASYMMETRY (documented, not silently ignored): Model A's own
    training set was a random holdout (or your personal CSV) drawn from ALL
    private listings at the time it was trained — some rows now sitting in
    this function's held-out test set may have been part of THAT training
    run. Model B, by construction, never trains on its own test rows. This
    function cannot retroactively fix that without retraining Model A on a
    row set that excludes Model B's test rows, which would stop Model A from
    being "the existing baseline, reproducible as-is" — so this is the best
    achievable fairness (same evaluation rows, each model's own encoders)
    without compromising Model A's preservation requirement. Treat Model A's
    number here as a mild best-case estimate for that reason.
    """
    from training import registry
    from Features import prepare_features

    model_a = registry.get_latest(session, "private_csv")
    model_b = registry.get_latest(session, "private_traintest")
    if model_a is None or model_b is None:
        raise ValueError(
            "Both private_csv and private_traintest must be trained and registered "
            "first — run `python run.py train_private_csv` and `python run.py "
            "train_private_traintest`."
        )

    test_df = build_private_holdout(session, test_fraction=test_fraction)
    if test_df.empty:
        raise ValueError("Held-out private test set is empty — not enough private listings yet.")

    rows = []
    for label, artifact in (("private_csv (Model A)", model_a), ("private_traintest (Model B)", model_b)):
        with open(artifact.encoder_path, "rb") as f:
            encoders = pickle.load(f)
        X, y_true, _ = prepare_features(test_df.copy(), label_encoders=encoders, fit=False)
        metrics = score_saved_model(artifact.artifact_path, artifact.encoder_path, X, y_true)
        rows.append({"model_name": f"{label} v{artifact.version}", **metrics})

    generate_comparison_report(rows, output_path=output_path)
    return rows


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

_REPORT_COLUMNS = [
    ("model_name", "Model"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("r2", "R²"),
    ("median_abs_error", "Median Error"),
    ("mape", "MAPE"),
    ("bias", "Bias"),
    ("pct_within_500", "±$500"),
    ("pct_within_1000", "±$1k"),
    ("pct_within_2500", "±$2.5k"),
    ("pct_within_5000", "±$5k"),
]


def _fmt(key, value):
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if key == "model_name":
        return str(value)
    if key in ("mae", "rmse", "median_abs_error", "bias"):
        return f"${value:,.0f}"
    if key == "r2":
        return f"{value:.3f}"
    if key == "mape":
        return f"{value:.1f}%"
    if key.startswith("pct_within"):
        return f"{value * 100:.1f}%"
    return str(value)


def generate_comparison_report(rows: list[dict], output_path: str = "reports/model_comparison.md") -> str:
    """
    `rows` is a list of {"model_name": str, **metrics_dict} (metrics_dict
    being whatever training.metrics.compute_metrics() / score_saved_model()
    returned, merged in). Writes a markdown table to `output_path` (creating
    the parent directory if missing), prints it, and returns the markdown
    string.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "| " + " | ".join(label for _, label in _REPORT_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _REPORT_COLUMNS) + " |"

    lines = [header, separator]
    for row in rows:
        cells = [_fmt(key, row.get(key)) for key, _ in _REPORT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join(lines) + "\n"

    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return markdown


# ---------------------------------------------------------------------------
# Plots (best-effort — a plotting failure must never crash the pipeline)
# ---------------------------------------------------------------------------

def _safe_plot(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning("Plot function %s failed; continuing without it.", fn.__name__, exc_info=True)
            return None
    return wrapper


@_safe_plot
def plot_actual_vs_predicted(y_true, y_pred, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Actual vs. Predicted")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


@_safe_plot
def plot_residuals(y_true, y_pred, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, residuals, alpha=0.5, s=15)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Residual (predicted - actual, $)")
    ax.set_title("Residuals vs. Actual")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


@_safe_plot
def plot_error_distribution(y_true, y_pred, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_pred - y_true)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(abs_errors, bins=30, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Absolute error ($)")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


@_safe_plot
def plot_error_by_bucket(df: pd.DataFrame, bucket_col: str, out_path, y_true_col="y_true", y_pred_col="y_pred", n_bins: int = 5):
    """
    Reusable for price range / vehicle age / mileage / condition: buckets
    `bucket_col` into `n_bins` quantile bins (falling back to equal-width
    bins if the column has too few unique values to quantile-cut) and plots
    mean absolute error per bucket. If `bucket_col` is already categorical
    (e.g. condition_overall values like "excellent"/"good"/"fair"/"poor",
    a string column rather than a continuous one), it's grouped directly on
    its existing values instead of being numerically binned.

    Expects `df` to have `y_true_col` and `y_pred_col` columns alongside
    `bucket_col`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["_abs_error"] = (df[y_pred_col] - df[y_true_col]).abs()

    if pd.api.types.is_numeric_dtype(df[bucket_col]):
        try:
            df["_bucket"] = pd.qcut(df[bucket_col], q=n_bins, duplicates="drop")
        except (ValueError, TypeError):
            df["_bucket"] = pd.cut(df[bucket_col], bins=n_bins)
    else:
        # Already-discrete/categorical column (e.g. condition labels) —
        # group directly rather than trying to numerically bin strings.
        df["_bucket"] = df[bucket_col]

    grouped = df.groupby("_bucket", observed=True)["_abs_error"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(b) for b in grouped.index], grouped.values)
    ax.set_xlabel(bucket_col)
    ax.set_ylabel("Mean absolute error ($)")
    ax.set_title(f"Mean Absolute Error by {bucket_col}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


@_safe_plot
def plot_error_over_time(df: pd.DataFrame, date_col: str, out_path, y_true_col="y_true", y_pred_col="y_pred"):
    """
    Plots mean absolute error over time, bucketed by day (based on
    `date_col`, e.g. first_seen). Expects `df` to have `y_true_col` and
    `y_pred_col` columns alongside `date_col`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["_abs_error"] = (df[y_pred_col] - df[y_true_col]).abs()
    df["_date"] = pd.to_datetime(df[date_col]).dt.date

    grouped = df.groupby("_date")["_abs_error"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped.index.astype(str), grouped.values, marker="o", markersize=3)
    ax.set_xlabel(date_col)
    ax.set_ylabel("Mean absolute error ($)")
    ax.set_title("Mean Absolute Error Over Time")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
