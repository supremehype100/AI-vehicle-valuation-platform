from __future__ import annotations

"""
training/registry.py

Thin wrapper around the `model_artifacts` table (see `ModelArtifact` in
models.py). Every training pipeline in this package registers one row here
per trained artifact.

NOTE: there is no live Postgres instance reachable in this sandbox, so the
functions below are import-clean and eyeballed-correct (straightforward
SQLAlchemy 2.0 `select()`/session usage mirroring the pattern already used
in `Features.load_training_data()` and `Model.py`'s `_find_comparables()`),
but are NOT exercised against a real database by this package's test suite.
DB integration is unverified here — flagged explicitly in the task report.
"""

import math
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ModelArtifact


def _json_safe(value):
    """Replace NaN/inf floats with None, recursively. Postgres JSONB rejects
    the bare NaN token Python's json module would otherwise emit (e.g. r2 on
    a constant-target validation set)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def register_model(
    session: Session,
    *,
    model_key: str,
    artifact_path: str,
    version: str | None = None,
    is_baseline: bool = False,
    encoder_path: str | None = None,
    meta_path: str | None = None,
    dataset_listing_type: str | None = None,
    dataset_row_count: int | None = None,
    dataset_fingerprint: str | None = None,
    split_strategy: str | None = None,
    feature_list: list | None = None,
    model_config: dict | None = None,
    metrics: dict | None = None,
    notes: str | None = None,
) -> ModelArtifact:
    """
    Insert a new row into model_artifacts and return it.

    `version` defaults to a UTC timestamp string ("%Y%m%d-%H%M%S") if the
    caller doesn't supply one, so every call produces a distinct, sortable
    version by default.
    """
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    artifact = ModelArtifact(
        model_key=model_key,
        version=version,
        is_baseline=is_baseline,
        artifact_path=artifact_path,
        encoder_path=encoder_path,
        meta_path=meta_path,
        dataset_listing_type=dataset_listing_type,
        dataset_row_count=dataset_row_count,
        dataset_fingerprint=dataset_fingerprint,
        split_strategy=split_strategy,
        feature_list=list(feature_list) if feature_list is not None else [],
        model_config=dict(model_config) if model_config is not None else {},
        metrics=_json_safe(metrics) if metrics is not None else {},
        notes=notes,
    )
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def get_latest(session: Session, model_key: str) -> ModelArtifact | None:
    """Return the most recently trained ModelArtifact for `model_key`, or None."""
    return session.execute(
        select(ModelArtifact)
        .where(ModelArtifact.model_key == model_key)
        .order_by(ModelArtifact.trained_at.desc())
        .limit(1)
    ).scalars().first()


def list_versions(session: Session, model_key: str) -> list[ModelArtifact]:
    """Return all ModelArtifact rows for `model_key`, newest first."""
    return session.execute(
        select(ModelArtifact)
        .where(ModelArtifact.model_key == model_key)
        .order_by(ModelArtifact.trained_at.desc())
    ).scalars().all()
