from __future__ import annotations

"""
training/_traintest_common.py

Shared implementation behind Model B (private_traintest) and Model C
(dealer_traintest) — same architecture, same feature pipeline, different
`listing_type` filter. Not part of the public API surface described in the
task (the task names private_traintest.py / dealer_traintest.py as the
pipelines); this module exists purely so those two files don't duplicate
~150 lines of identical logic. Leading underscore signals "internal".

Pipeline, mirroring the task's required shape:

  1. Query listings for the given listing_type, mirroring the exact filter
     Features.load_training_data() applies (price bounds, placeholder-price
     exclusion, year not null, listing_type) so the dataset here is
     comparable to what Model A trains on, modulo the split strategy.
  2. dedupe_by_fingerprint -> time_aware_split -> enforce_vin_disjoint ->
     assert_no_vin_leakage to carve out a held-out TEST set that never
     leaks into training (see training/splits.py docstring for why).
  3. A SECOND time-aware split of the remaining TRAIN rows carves out an
     internal early-stopping validation slice. This matters: XGBoost's
     early_stopping_rounds needs an eval_set, and using the real held-out
     TEST set for that would leak test information into model selection
     (the number of boosting rounds actually used would be chosen to
     minimize error on the "held out" set). Using a further time-sliced
     validation split from *within* train keeps the reported test metrics
     honest. Model.py's baseline doesn't need this distinction because it
     uses a single random holdout either way (or a personal CSV) —
     here we control for both the split strategy AND for test-set purity.
  4. Features.listings_to_dataframe() builds each split's DataFrame
     (reused, not reimplemented). leakage.zero_out_leaky_features()
     neutralizes days_on_market on all three splits before encoding.
  5. Features.prepare_features() fits label encoders on the TRAIN-CORE
     split only and reuses them (unseen categories fall back to the
     "unknown" bucket prepare_features() already reserves for exactly this
     situation) for the validation and test splits.
  6. An xgb.XGBRegressor with the same hyperparameters as Model.py's
     baseline (for an apples-to-apples architectural comparison) is
     trained with early stopping against the validation split, then
     scored ONCE against the untouched test split via
     training.metrics.compute_metrics.
  7. Model + encoders + meta are pickled under
     config.MODEL_DIR / <model_key> / <version>_{model,encoders,meta}.pkl
     and registered via training.registry.register_model().
"""

import logging
import pickle
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from Features import listings_to_dataframe, prepare_features, PLACEHOLDER_PRICES
from models import Listing
from training import splits, leakage, registry
from training.metrics import compute_metrics
from training.fingerprint import compute_dataset_fingerprint

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 50  # mirrors Model.py's MIN_TRAINING_SAMPLES guard

# Same architecture as Model.py's train() for a fair architectural
# comparison between the random-holdout baseline and the time-aware /
# VIN-disjoint split used here.
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
INTERNAL_VAL_FRACTION = 0.15  # carved out of TRAIN only, for early stopping


def _query_listings(session: Session, listing_type: str):
    """Mirrors Features.load_training_data()'s filter exactly (price bounds,
    placeholder-price exclusion, year not null, listing_type), so this
    dataset is apples-to-apples with what Model A trains on."""
    conditions = [
        Listing.price >= 1000,
        Listing.price <= 200_000,
        ~Listing.price.in_(PLACEHOLDER_PRICES),
        Listing.year.isnot(None),
        Listing.listing_type == listing_type,
    ]
    return session.execute(select(Listing).where(*conditions)).scalars().all()


def run_traintest(
    session: Session,
    *,
    listing_type: str,
    model_key: str,
    test_fraction: float = 0.15,
    min_training_samples: int = MIN_TRAINING_SAMPLES,
) -> dict:
    logger.info("Loading %s listings for %s...", listing_type, model_key)
    listings = _query_listings(session, listing_type)

    listings = splits.dedupe_by_fingerprint(listings)
    train_listings, test_listings = splits.time_aware_split(listings, test_fraction=test_fraction)
    train_listings, test_listings = splits.enforce_vin_disjoint(train_listings, test_listings)
    splits.assert_no_vin_leakage(train_listings, test_listings)

    if len(train_listings) < min_training_samples:
        raise ValueError(
            f"[{model_key}] Only {len(train_listings)} training rows for "
            f"listing_type={listing_type!r} after split — need at least "
            f"{min_training_samples}. Scrape more {listing_type} listings "
            f"before training this model."
        )
    if len(test_listings) == 0:
        raise ValueError(
            f"[{model_key}] Test split for listing_type={listing_type!r} is "
            f"empty (test_fraction={test_fraction} too small for "
            f"{len(listings)} total rows, or all test rows were reassigned "
            f"to train by enforce_vin_disjoint). Cannot evaluate this model."
        )

    # Carve an internal early-stopping validation slice out of TRAIN only —
    # the real test_listings set must stay untouched by model selection.
    train_core_listings, val_listings = splits.time_aware_split(
        train_listings, test_fraction=INTERNAL_VAL_FRACTION
    )
    train_core_listings, val_listings = splits.enforce_vin_disjoint(train_core_listings, val_listings)
    splits.assert_no_vin_leakage(train_core_listings, val_listings)

    if len(train_core_listings) < min_training_samples:
        raise ValueError(
            f"[{model_key}] Only {len(train_core_listings)} training rows "
            f"remain after carving out the internal validation split — need "
            f"at least {min_training_samples}. Scrape more {listing_type} "
            f"listings, or reduce test_fraction."
        )
    if len(val_listings) == 0:
        raise ValueError(
            f"[{model_key}] Internal early-stopping validation split is "
            f"empty — cannot train with early stopping. Scrape more "
            f"{listing_type} listings."
        )

    train_df = leakage.zero_out_leaky_features(listings_to_dataframe(train_core_listings))
    val_df = leakage.zero_out_leaky_features(listings_to_dataframe(val_listings))
    test_df = leakage.zero_out_leaky_features(listings_to_dataframe(test_listings))

    if len(train_df) < min_training_samples:
        raise ValueError(
            f"[{model_key}] Only {len(train_df)} rows survived "
            f"listings_to_dataframe() price/description filtering — need "
            f"at least {min_training_samples}."
        )
    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"[{model_key}] Validation or test split became empty after "
            f"listings_to_dataframe() filtering (val={len(val_df)}, "
            f"test={len(test_df)})."
        )

    feature_names = [c for c in train_df.columns if c != "price"]

    # Fingerprints the exact set of listing rows (train+val+test combined)
    # used by this run, so two runs with the same row COUNT but different
    # underlying data (e.g. after new scrapes) are distinguishable in the
    # registry.
    dataset_fingerprint = compute_dataset_fingerprint(
        [l.id for l in (train_core_listings + val_listings + test_listings)]
    )

    X_train, y_train, encoders = prepare_features(train_df, fit=True)
    X_val, y_val, _ = prepare_features(val_df, label_encoders=encoders, fit=False)
    X_test, y_test, _ = prepare_features(test_df, label_encoders=encoders, fit=False)

    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    metrics_dict = compute_metrics(y_test, preds)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = config.MODEL_DIR / model_key
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

    logger.info(
        "[%s] test MAE=$%.0f R2=%.3f (train_core=%d val=%d test=%d)",
        model_key, metrics_dict["mae"], metrics_dict["r2"],
        len(train_df), len(val_df), len(test_df),
    )

    artifact = registry.register_model(
        session,
        model_key=model_key,
        version=version,
        is_baseline=False,
        artifact_path=str(artifact_path),
        encoder_path=str(encoder_path),
        meta_path=str(meta_path),
        dataset_listing_type=listing_type,
        dataset_row_count=meta["total_samples"],
        dataset_fingerprint=dataset_fingerprint,
        split_strategy="time_aware_vin_grouped",
        feature_list=feature_names,
        model_config={**XGB_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        metrics=metrics_dict,
        notes=(
            f"Test split held out via time_aware_split (test_fraction="
            f"{test_fraction}) + enforce_vin_disjoint; internal "
            f"early-stopping validation carved separately from train only "
            f"(INTERNAL_VAL_FRACTION={INTERNAL_VAL_FRACTION}) so test "
            f"metrics are not influenced by early-stopping model selection."
        ),
    )

    return {"meta": meta, "artifact": artifact, "model": model, "encoders": encoders}
