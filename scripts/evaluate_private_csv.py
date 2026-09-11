"""
scripts/evaluate_private_csv.py

Runs the currently-saved model against YOUR local personal validation CSV and
prints an AGGREGATE, sanitized summary — row counts, column names/dtypes,
missing-value percentages, numeric ranges, and prediction metrics (MAE, RMSE,
R², median absolute error, MAPE, bias, percent within $500/$1k/$2.5k/$5k).

It never prints individual rows, descriptions, or any other per-record data —
only aggregates safe to paste back into a conversation. The CSV itself never
leaves your machine; this script only reads it locally.

Usage:
    python scripts/evaluate_private_csv.py --csv /private/path/validation.csv
    python scripts/evaluate_private_csv.py --csv validation.csv --model-dir models/private_traintest

Expected CSV columns (same shape Model.py's existing personal-CSV validation
path already expects): sell_price, mileage, year, make, model,
original_description (optional but improves feature quality — used only for
description-flag extraction, never printed).
"""
from __future__ import annotations
import argparse
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Model import _load_csv_for_validation  # reuse existing CSV-shape parsing, don't duplicate it
from Features import prepare_features
from training.metrics import compute_metrics


def _column_summary(df: pd.DataFrame) -> list[dict]:
    summary = []
    for col in df.columns:
        s = df[col]
        entry = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            entry["min"] = float(s.min())
            entry["max"] = float(s.max())
            entry["mean"] = round(float(s.mean()), 2)
        summary.append(entry)
    return summary


def load_model_artifact(model_dir: Path):
    model_path = model_dir / "xgb_price_model.pkl"
    encoder_path = model_dir / "label_encoders.pkl"
    meta_path = model_dir / "model_meta.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run `python run.py train` (or a training/*.py pipeline) first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    return model, encoders, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to your local personal validation CSV (never uploaded).")
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory containing xgb_price_model.pkl / label_encoders.pkl (default: models/, i.e. the "
             "current Model A baseline). Point this at models/private_traintest/<version>/ to evaluate Model B instead.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # --- Raw CSV shape summary (aggregate only) ---
    raw_df = pd.read_csv(csv_path)
    print("=== CSV summary ===")
    print(f"Rows: {len(raw_df)}")
    print(f"Columns: {list(raw_df.columns)}")
    for entry in _column_summary(raw_df):
        print(f"  {entry}")

    # --- Parse into the model's expected feature shape (reuses Model.py's own logic) ---
    df = _load_csv_for_validation(str(csv_path))
    if df.empty:
        print("\nNo usable rows after cleaning (need sell_price >= 100 and a valid year). Nothing to evaluate.")
        sys.exit(0)

    print(f"\nUsable rows after cleaning: {len(df)}")

    model_dir = Path(args.model_dir)
    model, encoders, meta = load_model_artifact(model_dir)
    if meta:
        print(f"\nEvaluating model: version={meta.get('model_version', 'unknown')}, "
              f"trained_on={meta.get('training_samples', 'unknown')} listings")

    X, y_true, _ = prepare_features(df, label_encoders=encoders, fit=False)
    y_pred = model.predict(X)

    metrics = compute_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))

    print("\n=== Prediction metrics (aggregate only — no individual rows shown) ===")
    print(json.dumps(metrics, indent=2))
    print("\nSafe to paste this whole output back into the conversation.")


if __name__ == "__main__":
    main()
