from __future__ import annotations

"""
training/salvage_traintest.py

Model D — salvage/auction vehicles. Deliberately does NOT reuse
Features.listings_to_dataframe(), which assumes retail-listing semantics
(seller_type, body_type/drivetrain/fuel_type from raw_json, description-flag
extraction). Salvage vehicles price very differently — auction/wholesale
dynamics dominate (buyer pool is dealers/rebuilders, not retail
consumers), and damage severity is usually the single biggest price driver,
which retail listings don't even have columns for. So this module builds
its own minimal salvage-specific feature frame using:

    year, make, model, mileage, damage_primary, damage_secondary,
    title_type, run_drive_status, auction_house

as inputs and `price` as the target.

WHAT `price` MEANS FOR SALVAGE ROWS
------------------------------------
For listing_type="salvage" rows, `price` represents the auction sale/bid
amount, not a retail asking price. `Listing.auction_sale_status` tells you
whether that number is final: "sold" means `price` is the actual hammer
price (a true sale, the best kind of label); "unsold" or "pending" means
`price` is only an estimate/current bid, not a completed transaction. This
module only trains on rows where `auction_sale_status == "sold"` — training
on live/estimated bid amounts as if they were ground-truth sale prices would
teach the model on labels that were never actually realized.

LABEL ENCODING PATTERN
------------------------
This module copies the *pattern* Features.prepare_features() uses
(per-column sklearn LabelEncoder, always include an "unknown" bucket in the
fitted vocabulary, map unseen categories to "unknown" at transform time) but
NOT its retail-specific column list — salvage has entirely different
categorical columns (damage_primary, title_type, etc. instead of
body_type/drivetrain/fuel_type/transmission).

NO DATA YET
------------
The stat.vin scraper that will populate listing_type="salvage" rows is being
built by a different agent concurrently and has not run against production
yet, so this pipeline will find zero rows if run today. `run()` fails with a
clear, friendly ValueError in that case rather than crashing obscurely deep
inside pandas/sklearn on an empty DataFrame. This module is structurally
complete but genuinely untested against real data, because none exists yet.
"""

import logging
import pickle
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from models import Listing
from training import splits, registry
from training.metrics import compute_metrics
from training.fingerprint import compute_dataset_fingerprint

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 50
INTERNAL_VAL_FRACTION = 0.15

CAT_COLS = [
    "make", "model", "damage_primary", "damage_secondary",
    "title_type", "run_drive_status", "auction_house",
]

# Same architecture as the other pipelines' XGBRegressor, for consistency.
XGB_PARAMS = dict(
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
    eval_metric="mae",
)
EARLY_STOPPING_ROUNDS = 30


def salvage_listings_to_dataframe(listings) -> pd.DataFrame:
    """Build the salvage-specific feature frame. Only rows with a completed
    sale (auction_sale_status == 'sold') are included — see module docstring."""
    rows = []
    for l in listings:
        if l.price is None or l.price <= 0 or l.year is None:
            continue
        if getattr(l, "auction_sale_status", None) != "sold":
            continue

        rows.append({
            "price": l.price,
            "year": l.year,
            "make": (l.make or "unknown").lower().strip(),
            "model": (l.model or "unknown").lower().strip(),
            "mileage": l.mileage if l.mileage is not None else 100_000,
            "damage_primary": (getattr(l, "damage_primary", None) or "unknown").lower().strip(),
            "damage_secondary": (getattr(l, "damage_secondary", None) or "unknown").lower().strip(),
            "title_type": (getattr(l, "title_type", None) or "unknown").lower().strip(),
            "run_drive_status": (getattr(l, "run_drive_status", None) or "unknown").lower().strip(),
            "auction_house": (getattr(l, "auction_house", None) or "unknown").lower().strip(),
        })
    return pd.DataFrame(rows)


def prepare_salvage_features(df: pd.DataFrame, label_encoders: dict | None = None, fit: bool = False):
    """Same per-column LabelEncoder pattern as Features.prepare_features(),
    applied to the salvage-specific CAT_COLS instead of the retail column
    list. Returns (X: np.ndarray, y: np.ndarray, label_encoders: dict)."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    target = df.pop("price")

    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "unknown"

        if fit:
            le = LabelEncoder()
            values = df[col].astype(str).tolist() + ["unknown"]
            le.fit(values)
            df[col] = le.transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "unknown")
            df[col] = le.transform(df[col].astype(str))

    return df.values, target.values, label_encoders


def _query_listings(session: Session):
    conditions = [
        Listing.listing_type == config.LISTING_TYPE_SALVAGE,
        Listing.price.isnot(None),
        Listing.price > 0,
        Listing.year.isnot(None),
    ]
    return session.execute(select(Listing).where(*conditions)).scalars().all()


def run(session: Session, test_fraction: float = 0.15, min_training_samples: int = MIN_TRAINING_SAMPLES) -> dict:
    """
    Train Model D (salvage) and register the resulting artifact in
    model_artifacts.

    Raises a friendly ValueError (not a raw crash) if no salvage listings
    exist yet — expected today, since the stat.vin scraper hasn't run
    against production.
    """
    logger.info("Loading salvage listings...")
    listings = _query_listings(session)

    if not listings:
        raise ValueError(
            "No salvage listings found — run the stat.vin scraper first, "
            "then re-run this pipeline. (listing_type='salvage' has zero "
            "rows in the database.)"
        )

    listings = splits.dedupe_by_fingerprint(listings)
    train_listings, test_listings = splits.time_aware_split(listings, test_fraction=test_fraction)
    train_listings, test_listings = splits.enforce_vin_disjoint(train_listings, test_listings)
    splits.assert_no_vin_leakage(train_listings, test_listings)

    train_core_listings, val_listings = splits.time_aware_split(
        train_listings, test_fraction=INTERNAL_VAL_FRACTION
    )
    train_core_listings, val_listings = splits.enforce_vin_disjoint(train_core_listings, val_listings)
    splits.assert_no_vin_leakage(train_core_listings, val_listings)

    train_df = salvage_listings_to_dataframe(train_core_listings)
    val_df = salvage_listings_to_dataframe(val_listings)
    test_df = salvage_listings_to_dataframe(test_listings)

    if len(train_df) < min_training_samples:
        raise ValueError(
            f"[salvage_traintest] Only {len(train_df)} sold salvage rows "
            f"available for training after filtering to "
            f"auction_sale_status=='sold' — need at least "
            f"{min_training_samples}. Run the stat.vin scraper first, or "
            f"wait for more completed auctions."
        )
    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"[salvage_traintest] Validation or test split is empty after "
            f"filtering (val={len(val_df)}, test={len(test_df)}). Not "
            f"enough sold salvage data yet to hold out a split."
        )

    feature_names = [c for c in train_df.columns if c != "price"]

    dataset_fingerprint = compute_dataset_fingerprint(
        [l.id for l in (train_core_listings + val_listings + test_listings)]
    )

    X_train, y_train, encoders = prepare_salvage_features(train_df, fit=True)
    X_val, y_val, _ = prepare_salvage_features(val_df, label_encoders=encoders, fit=False)
    X_test, y_test, _ = prepare_salvage_features(test_df, label_encoders=encoders, fit=False)

    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    metrics_dict = compute_metrics(y_test, preds)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = config.MODEL_DIR / "salvage_traintest"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / f"{version}_model.pkl"
    encoder_path = out_dir / f"{version}_encoders.pkl"
    meta_path = out_dir / f"{version}_meta.pkl"

    meta = {
        "train_core_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "feature_names": feature_names,
        **metrics_dict,
    }

    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    artifact = registry.register_model(
        session,
        model_key="salvage_traintest",
        version=version,
        is_baseline=False,
        artifact_path=str(artifact_path),
        encoder_path=str(encoder_path),
        meta_path=str(meta_path),
        dataset_listing_type="salvage",
        dataset_row_count=meta["total_samples"],
        dataset_fingerprint=dataset_fingerprint,
        split_strategy="time_aware_vin_grouped",
        feature_list=feature_names,
        model_config={**XGB_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        metrics=metrics_dict,
        notes=(
            "Trained only on auction_sale_status=='sold' rows (price is a "
            "final hammer price, not an estimate/live bid). Custom salvage "
            "feature set — does not reuse Features.listings_to_dataframe()."
        ),
    )

    return {"meta": meta, "artifact": artifact, "model": model, "encoders": encoders}
