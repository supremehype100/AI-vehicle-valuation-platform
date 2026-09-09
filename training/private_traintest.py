from __future__ import annotations

"""
training/private_traintest.py

Model B — same architecture as Model A (private_csv / Model.py's baseline)
but with a leakage-aware time-aware/VIN-disjoint train/test split instead of
a random holdout, and the days_on_market leak neutralized (see
training/leakage.py). Trained on `listing_type="private"` listings, same as
Model A, so the two are directly comparable in training/evaluate.py's
report — the only thing that differs is the split strategy and leak
handling.

Shared implementation lives in training/_traintest_common.py (identical
logic to dealer_traintest.py, parameterized by listing_type/model_key).
"""

from sqlalchemy.orm import Session

from training._traintest_common import run_traintest
from config import LISTING_TYPE_PRIVATE


def run(session: Session, test_fraction: float = 0.15) -> dict:
    """
    Train Model B and register the resulting artifact in model_artifacts.

    Returns {"meta": dict, "artifact": ModelArtifact, "model": XGBRegressor,
    "encoders": dict}.
    """
    return run_traintest(
        session,
        listing_type=LISTING_TYPE_PRIVATE,
        model_key="private_traintest",
        test_fraction=test_fraction,
    )
