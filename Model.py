from __future__ import annotations
"""
ml/model.py

Trains an XGBoost price prediction model on scraped listings
and provides a valuation function.

Usage:
    # Train
    train(session)

    # Predict
    result = valuate(session, make="toyota", model="camry", year=2019, mileage=80000)
    print(result)
    # {
    #   "predicted_price": 18500,
    #   "confidence_low": 16200,
    #   "confidence_high": 20800,
    #   "comparable_listings": [...],
    #   "model_version": "xgb_v1",
    #   "training_samples": 412,
    # }
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from Features import load_training_data, prepare_features, extract_description_flags, extract_transmission
from models import Listing, PricePrediction, TrainingRun

def _load_csv_for_validation(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rows = []
    for _, row in df.iterrows():
        sell = float(row["sell_price"]) if pd.notna(row.get("sell_price")) else None
        if sell is None or sell < 100:
            continue

        mileage = int(row["mileage"]) if pd.notna(row.get("mileage")) else 100_000
        year = int(row["year"]) if pd.notna(row.get("year")) else None
        if year is None:
            continue

        desc = str(row.get("original_description", ""))
        desc_flags = extract_description_flags(desc)

        rows.append({
            "price": sell,
            "year": year,
            "mileage": mileage,
            "days_on_market": 0,
            "make": str(row.get("make", "unknown")).lower().strip(),
            "model": str(row.get("model", "unknown")).lower().split()[0],
            "trim": "unknown",
            "seller_type": "private",
            "body_type": "unknown",
            "drivetrain": "unknown",
            "fuel_type": "unknown",
            "transmission": extract_transmission(desc),
            **desc_flags,
        })

    return pd.DataFrame(rows)


logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_price_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"
META_PATH = MODEL_DIR / "model_meta.pkl"

MODEL_VERSION = "xgb_v1"
MIN_TRAINING_SAMPLES = 50  # won't train with fewer than this


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(session: Session, cv_csv_path: str | None = None) -> dict:
    """
    Train XGBoost on all listings in the DB.
    Saves model + encoders to disk.
    Returns training metrics.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    logger.info("Loading training data...")
    df_train = load_training_data(session)

    if len(df_train) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(df_train)} samples — need at least {MIN_TRAINING_SAMPLES}."
        )

    X_train, y_train, encoders = prepare_features(df_train, fit=True)

    if cv_csv_path:
        df_val = _load_csv_for_validation(cv_csv_path)
        logger.info("Training on %d listings, validating on %d personal records", len(df_train), len(df_val))
        X_val, y_val, _ = prepare_features(df_val, label_encoders=encoders, fit=False)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        logger.info("Training on %d listings", len(df_train))

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=17,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    median_pct_err = float(np.median(np.abs(val_preds - y_val) / y_val) * 100)

    meta = {
        "training_samples": len(df_train),
        "val_samples": len(y_val),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "median_pct_error": round(median_pct_err, 2),
        "model_version": MODEL_VERSION,
        "feature_names": list(df_train.drop(columns=["price"]).columns) if "price" in df_train.columns else [],
    }

    # Save everything
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    logger.info(
        "Training complete. MAE=$%.0f | R²=%.3f | Median error=%.1f%%",
        mae, r2, median_pct_err
    )
    session.add(
        TrainingRun(
            training_samples=meta["training_samples"],
            val_samples=meta["val_samples"],
            mae=meta["mae"],
            r2=meta["r2"],
            median_pct_error=meta["median_pct_error"],
            model_version=meta["model_version"],
            used_personal_csv=cv_csv_path is not None,
        )
    )

    session.commit()
    return meta


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train() first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return model, encoders, meta


# ---------------------------------------------------------------------------
# Valuate
# ---------------------------------------------------------------------------

def valuate(
    session: Session,
    make: str,
    model_name: str,
    year: int,
    mileage: int,
    trim: str | None = None,
    seller_type: str = "private",
    description: str | None = None,
    body_type: str | None = None,
    drivetrain: str | None = None,
    fuel_type: str | None = None,
    n_comparables: int = 5,
) -> dict:
    """
    Predict the market price for a vehicle.

    Returns predicted price, confidence range, and comparable active listings.
    """
    xgb_model, encoders, meta = _load_model()

    desc_flags = extract_description_flags(description)

    # Build a single-row DataFrame matching training features exactly
    row = {
        "year": year,
        "mileage": mileage,
        "days_on_market": 0,
        "make": make.lower().strip(),
        "model": model_name.lower().strip(),
        "trim": (trim or "unknown").lower().strip(),
        "seller_type": seller_type,
        "body_type": (body_type or "unknown").lower().strip(),
        "drivetrain": (drivetrain or "unknown").lower().strip(),
        "fuel_type": (fuel_type or "unknown").lower().strip(),
        "transmission": "unknown",
        **desc_flags,
    }

    df = pd.DataFrame([row])

    # Add dummy price column so prepare_features doesn't break
    df["price"] = 0
    X, _, _ = prepare_features(df, label_encoders=encoders, fit=False)

    predicted = float(xgb_model.predict(X)[0])

    # Confidence interval via quantile estimation on training residuals
    # Simple approach: use ±20% as a starting interval, tightened by model confidence
    # A proper approach is to train quantile regression models separately — add later
    confidence_low = predicted * 0.85
    confidence_high = predicted * 1.15

    # Find comparable active listings from DB
    comparables = _find_comparables(session, make, model_name, year, mileage, n_comparables)

    # If we have comparables, use their price distribution to refine the interval
    if comparables:
        comp_prices = [c["price"] for c in comparables if c["price"]]
        if len(comp_prices) >= 3:
            confidence_low = max(confidence_low, float(np.percentile(comp_prices, 15)))
            confidence_high = min(confidence_high, float(np.percentile(comp_prices, 85)))

    return {
        "predicted_price": round(predicted, 2),
        "confidence_low": round(confidence_low, 2),
        "confidence_high": round(confidence_high, 2),
        "comparable_listings": comparables,
        "model_version": meta["model_version"],
        "training_samples": meta["training_samples"],
        "model_mae": meta["mae"],
    }


def _find_comparables(
    session: Session,
    make: str,
    model_name: str,
    year: int,
    mileage: int,
    n: int = 5,
) -> list[dict]:
    """
    Find similar active listings from the DB.
    Matches on make/model, ±2 years, mileage within 30k.
    """
    rows = session.execute(
        select(Listing).where(
            Listing.make.ilike(make),
            Listing.model.ilike(model_name),
            Listing.year.between(year - 2, year + 2),
            Listing.mileage.between(max(0, mileage - 30_000), mileage + 30_000),
            Listing.price.isnot(None),
            Listing.status == "active",
        ).order_by(
            # Rank by how close year and mileage are to the query
            (Listing.year - year) * (Listing.year - year) +
            ((Listing.mileage - mileage) / 10_000) * ((Listing.mileage - mileage) / 10_000)
        ).limit(n)
    ).scalars().all()

    return [
        {
            "source_id": r.source_id,
            "title": r.title,
            "price": r.price,
            "year": r.year,
            "mileage": r.mileage,
            "location": r.location,
            "url": r.url,
            "days_on_market": r.days_on_market,
        }
        for r in rows
    ]


def score_listing(session: Session, listing: Listing) -> dict | None:
    """
    Run valuation on a single listing and store the result in price_predictions.
    Returns the prediction dict, or None if model isn't trained.
    """
    try:
        result = valuate(
            session,
            make=listing.make or "",
            model_name=listing.model or "",
            year=listing.year or 2010,
            mileage=listing.mileage or 100_000,
            trim=listing.trim,
            seller_type=listing.seller_type or "private",
            description=listing.description,
            body_type=(listing.raw_json or {}).get("body_type"),
            drivetrain=(listing.raw_json or {}).get("drivetrain"),
            fuel_type=(listing.raw_json or {}).get("fuel_type"),
        )
    except FileNotFoundError:
        return None  # model not trained yet

    predicted = result["predicted_price"]
    delta_pct = ((listing.price - predicted) / predicted) if (listing.price and predicted) else None

    # Upsert prediction row
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(PricePrediction).values(
        listing_id=listing.id,
        predicted_price=predicted,
        confidence_low=result["confidence_low"],
        confidence_high=result["confidence_high"],
        price_delta_pct=delta_pct,
        model_version=result["model_version"],
    ).on_conflict_do_update(
        constraint="uq_prediction_listing_model",
        set_={
            "predicted_price": predicted,
            "confidence_low": result["confidence_low"],
            "confidence_high": result["confidence_high"],
            "price_delta_pct": delta_pct,
        }
    )
    session.execute(stmt)

    return {**result, "price_delta_pct": delta_pct}