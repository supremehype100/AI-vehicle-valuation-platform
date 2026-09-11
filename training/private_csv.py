from __future__ import annotations

"""
training/private_csv.py

Model A — the existing, untouched baseline. This module does NOT
reimplement any training logic; it calls `Model.train()` (finalized,
off-limits to edit) exactly as before, and adds model-registry bookkeeping
on top so the baseline shows up in `model_artifacts` alongside the newer
pipelines (private_traintest / dealer_traintest / salvage_traintest).

`Model.train()` already benefits from the `listing_type="private"` default
filter baked into `Features.load_training_data()` (an intentional,
documented bug fix already applied at that layer — dealer listings used to
leak into what was supposed to be a private-party-only baseline; now they're
excluded). No code changes were needed here to pick that up.
"""

from sqlalchemy.orm import Session

import Model
from training import registry
from training.fingerprint import compute_proxy_fingerprint


def run(session: Session, cv_csv_path: str | None = None) -> dict:
    """
    Train Model A exactly as before (via the untouched Model.train()) and
    register the resulting artifact in model_artifacts.

    Returns {"meta": <dict returned by Model.train()>, "artifact": <ModelArtifact>}.
    """
    meta = Model.train(session, cv_csv_path=cv_csv_path)

    # Model.train() (finalized/off-limits) doesn't return the row IDs it
    # trained on, only aggregate counts — so unlike the *_traintest
    # pipelines (which fingerprint the exact listing IDs used), this is a
    # proxy fingerprint over available metadata. It still reliably detects
    # "did anything about this run's inputs change" (sample counts, feature
    # set, CSV path), just not a same-count-different-rows case.
    dataset_fingerprint = compute_proxy_fingerprint(
        meta["training_samples"],
        meta["val_samples"],
        sorted(meta.get("feature_names", [])),
        cv_csv_path or "random_holdout",
    )

    artifact = registry.register_model(
        session,
        model_key="private_csv",
        is_baseline=True,
        artifact_path=str(Model.MODEL_PATH),
        encoder_path=str(Model.ENCODER_PATH),
        meta_path=str(Model.META_PATH),
        dataset_listing_type="private",
        dataset_row_count=meta["training_samples"],
        dataset_fingerprint=dataset_fingerprint,
        split_strategy="personal_csv" if cv_csv_path else "random_holdout",
        feature_list=meta.get("feature_names", []),
        model_config={},
        metrics={k: meta[k] for k in ("mae", "r2", "median_pct_error")},
        notes="Model A baseline — unmodified Model.train(), wrapped only for registry bookkeeping.",
    )

    return {"meta": meta, "artifact": artifact}
