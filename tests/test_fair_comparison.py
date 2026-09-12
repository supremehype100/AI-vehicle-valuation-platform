"""
Tests for training/evaluate.py's compare_private_baseline_vs_traintest —
the fair, same-holdout comparison between Model A (private_csv) and Model B
(private_traintest). registry.get_latest / build_private_holdout are mocked
(no live DB here), but the pickle loading, per-model encoder application,
and scoring path are exercised for real.
"""
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from training.evaluate import compare_private_baseline_vs_traintest


class _FixedPredictModel:
    """Stand-in for an XGBRegressor — returns a caller-supplied prediction
    array (ignoring X's actual values beyond row count) so tests don't need
    a real trained booster, while still letting each row's prediction differ
    (avoiding a coincidental tie between two flat/constant predictors)."""
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        assert X.shape[0] == len(self.predictions)
        return self.predictions


def _make_holdout_df():
    return pd.DataFrame([
        {
            "price": 15000, "year": 2018, "mileage": 80000, "days_on_market": 0,
            "make": "toyota", "model": "camry", "trim": "unknown", "seller_type": "private",
            "body_type": "sedan", "drivetrain": "fwd", "fuel_type": "gas", "transmission": "automatic",
            "accident_free": 1, "one_owner": 0, "new_tires": 0, "new_brakes": 0, "recent_service": 0,
            "no_rust": 0, "immaculate": 0, "rebuilt_engine": 0, "rebuilt_transmission": 0,
            "rebuilt": 0, "salvage": 0, "needs_work": 0, "rust": 0, "leak": 0, "hail": 0,
            "misfire": 0, "inop": 0, "parts_car": 0,
        },
        {
            "price": 20000, "year": 2020, "mileage": 40000, "days_on_market": 0,
            "make": "honda", "model": "civic", "trim": "unknown", "seller_type": "private",
            "body_type": "sedan", "drivetrain": "fwd", "fuel_type": "gas", "transmission": "automatic",
            "accident_free": 0, "one_owner": 1, "new_tires": 0, "new_brakes": 0, "recent_service": 0,
            "no_rust": 0, "immaculate": 0, "rebuilt_engine": 0, "rebuilt_transmission": 0,
            "rebuilt": 0, "salvage": 0, "needs_work": 0, "rust": 0, "leak": 0, "hail": 0,
            "misfire": 0, "inop": 0, "parts_car": 0,
        },
    ])


def _pickle_encoders(path):
    cat_cols = ["make", "model", "trim", "seller_type", "body_type", "drivetrain", "fuel_type", "transmission"]
    vocab = {
        "make": ["toyota", "honda", "unknown"],
        "model": ["camry", "civic", "unknown"],
        "trim": ["unknown"],
        "seller_type": ["private", "unknown"],
        "body_type": ["sedan", "unknown"],
        "drivetrain": ["fwd", "unknown"],
        "fuel_type": ["gas", "unknown"],
        "transmission": ["automatic", "unknown"],
    }
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(vocab[col])
        encoders[col] = le
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def test_raises_when_model_a_not_registered():
    with patch("training.registry.get_latest", side_effect=[None, SimpleNamespace()]):
        with pytest.raises(ValueError, match="private_csv"):
            compare_private_baseline_vs_traintest(session=None)


def test_raises_when_model_b_not_registered():
    with patch("training.registry.get_latest", side_effect=[SimpleNamespace(), None]):
        with pytest.raises(ValueError, match="private_traintest"):
            compare_private_baseline_vs_traintest(session=None)


def test_raises_when_holdout_is_empty():
    fake_artifact = SimpleNamespace(version="v1", artifact_path="x", encoder_path="y")
    with patch("training.registry.get_latest", side_effect=[fake_artifact, fake_artifact]), \
         patch("training.evaluate.build_private_holdout", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="empty"):
            compare_private_baseline_vs_traintest(session=None)


def test_scores_both_models_on_the_same_holdout_with_their_own_encoders(tmp_path):
    encoder_path_a = tmp_path / "a_encoders.pkl"
    encoder_path_b = tmp_path / "b_encoders.pkl"
    model_path_a = tmp_path / "a_model.pkl"
    model_path_b = tmp_path / "b_model.pkl"
    report_path = tmp_path / "report.md"

    # Holdout truth is [15000, 20000]. Model A is way off on both rows;
    # Model B is close on both — an unambiguous MAE ordering, not a
    # coincidental tie between two flat predictors.
    _pickle_encoders(encoder_path_a)
    _pickle_encoders(encoder_path_b)
    with open(model_path_a, "wb") as f:
        pickle.dump(_FixedPredictModel(predictions=[9000.0, 12000.0]), f)   # errors: 6000, 8000 -> MAE 7000
    with open(model_path_b, "wb") as f:
        pickle.dump(_FixedPredictModel(predictions=[14800.0, 19800.0]), f)  # errors: 200, 200 -> MAE 200

    artifact_a = SimpleNamespace(version="20260101-000000", artifact_path=str(model_path_a), encoder_path=str(encoder_path_a))
    artifact_b = SimpleNamespace(version="20260102-000000", artifact_path=str(model_path_b), encoder_path=str(encoder_path_b))

    with patch("training.registry.get_latest", side_effect=[artifact_a, artifact_b]), \
         patch("training.evaluate.build_private_holdout", return_value=_make_holdout_df()):
        rows = compare_private_baseline_vs_traintest(session=None, output_path=str(report_path))

    assert len(rows) == 2
    assert rows[0]["model_name"].startswith("private_csv (Model A)")
    assert rows[1]["model_name"].startswith("private_traintest (Model B)")
    # Both scored against the identical two-row holdout — Model B (smaller
    # offset) must show a lower MAE than Model A on THIS SAME data.
    assert rows[1]["mae"] < rows[0]["mae"]
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "private_csv (Model A)" in report_text
    assert "private_traintest (Model B)" in report_text
