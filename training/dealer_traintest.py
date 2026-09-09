from __future__ import annotations

"""
training/dealer_traintest.py

Model C — identical architecture to private_traintest.py (Model B), trained
on `listing_type="dealer"` listings instead of private-party ones. Dealer
listings have their own price dynamics (dealer markup, reconditioning,
warranty inclusion) so this is intentionally a separate model rather than
mixed into the private dataset.

Shared implementation lives in training/_traintest_common.py (identical
logic to private_traintest.py, parameterized by listing_type/model_key).
"""

from sqlalchemy.orm import Session

from training._traintest_common import run_traintest
from config import LISTING_TYPE_DEALER


def run(session: Session, test_fraction: float = 0.15) -> dict:
    """
    Train Model C and register the resulting artifact in model_artifacts.

    Returns {"meta": dict, "artifact": ModelArtifact, "model": XGBRegressor,
    "encoders": dict}.
    """
    return run_traintest(
        session,
        listing_type=LISTING_TYPE_DEALER,
        model_key="dealer_traintest",
        test_fraction=test_fraction,
    )
