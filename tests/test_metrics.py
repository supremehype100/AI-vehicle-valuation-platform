"""
tests/test_metrics.py

Pure-function tests for training.metrics.compute_metrics — synthetic
arrays, hand-computed expected values. No DB, no network.
"""

import math

import pytest

from training.metrics import compute_metrics


def test_basic_metrics_hand_computed():
    y_true = [100, 200, 300, 400, 500]
    y_pred = [110, 190, 300, 420, 480]
    # errors (pred - true) = [10, -10, 0, 20, -20]
    # abs errors           = [10,  10, 0, 20,  20]

    result = compute_metrics(y_true, y_pred)

    assert result["mae"] == pytest.approx(12.0)
    assert result["rmse"] == pytest.approx(math.sqrt(200.0))  # mean(sq err)=200
    assert result["median_abs_error"] == pytest.approx(10.0)
    assert result["bias"] == pytest.approx(0.0)  # (10-10+0+20-20)/5

    # R^2: ss_res=1000, ss_tot=100000 -> 1 - 0.01 = 0.99
    assert result["r2"] == pytest.approx(0.99)

    # MAPE: (10/100 + 10/200 + 0/300 + 20/400 + 20/500)/5 * 100
    #     = (0.10 + 0.05 + 0 + 0.05 + 0.04)/5*100 = 4.8
    assert result["mape"] == pytest.approx(4.8)

    # All abs errors (0,10,10,20,20) are within every threshold.
    assert result["pct_within_500"] == pytest.approx(1.0)
    assert result["pct_within_1000"] == pytest.approx(1.0)
    assert result["pct_within_2500"] == pytest.approx(1.0)
    assert result["pct_within_5000"] == pytest.approx(1.0)

    assert result["n"] == 5


def test_error_distribution_percentiles_hand_computed():
    y_true = [100, 200, 300, 400, 500]
    y_pred = [110, 190, 300, 420, 480]
    # sorted abs errors: [0, 10, 10, 20, 20], n=5
    # numpy linear-interpolation percentile, idx = (n-1)*p/100:
    #   p10: idx=0.4 -> 0 + 0.4*(10-0) = 4.0
    #   p25: idx=1.0 -> 10.0
    #   p50: idx=2.0 -> 10.0
    #   p75: idx=3.0 -> 20.0
    #   p90: idx=3.6 -> 20 + 0.6*(20-20) = 20.0
    result = compute_metrics(y_true, y_pred)
    dist = result["error_distribution"]

    assert dist["p10"] == pytest.approx(4.0)
    assert dist["p25"] == pytest.approx(10.0)
    assert dist["p50"] == pytest.approx(10.0)
    assert dist["p75"] == pytest.approx(20.0)
    assert dist["p90"] == pytest.approx(20.0)


def test_pct_within_thresholds_with_larger_errors():
    y_true = [10_000, 10_000, 10_000, 10_000]
    y_pred = [10_400, 10_900, 12_000, 16_000]
    # abs errors: 400, 900, 2000, 6000

    result = compute_metrics(y_true, y_pred)

    assert result["pct_within_500"] == pytest.approx(1 / 4)   # only 400
    assert result["pct_within_1000"] == pytest.approx(2 / 4)  # 400, 900
    assert result["pct_within_2500"] == pytest.approx(3 / 4)  # 400, 900, 2000
    assert result["pct_within_5000"] == pytest.approx(3 / 4)  # 6000 excluded


def test_bias_direction():
    # model consistently OVER-predicts by 100
    y_true = [1000, 2000, 3000]
    y_pred = [1100, 2100, 3100]
    result = compute_metrics(y_true, y_pred)
    assert result["bias"] == pytest.approx(100.0)
    assert result["mae"] == pytest.approx(100.0)

    # model consistently UNDER-predicts by 100
    y_pred_under = [900, 1900, 2900]
    result_under = compute_metrics(y_true, y_pred_under)
    assert result_under["bias"] == pytest.approx(-100.0)


def test_perfect_predictions():
    y_true = [1000, 2000, 3000, 4000]
    y_pred = [1000, 2000, 3000, 4000]
    result = compute_metrics(y_true, y_pred)

    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)
    assert result["mape"] == pytest.approx(0.0)
    assert result["bias"] == pytest.approx(0.0)
    for key in ("pct_within_500", "pct_within_1000", "pct_within_2500", "pct_within_5000"):
        assert result[key] == pytest.approx(1.0)


def test_mape_guard_excludes_near_zero_true_values():
    # One y_true is 0 (or below epsilon) — MAPE must not raise or become inf,
    # and must be computed only over the remaining valid rows.
    y_true = [0, 1000, 2000]
    y_pred = [50, 1100, 1900]
    # valid rows (|y_true| > epsilon): 1000/1100 and 2000/1900
    # mape = mean(100/1000, 100/2000) * 100 = mean(0.10, 0.05)*100 = 7.5
    result = compute_metrics(y_true, y_pred)
    assert not math.isnan(result["mape"])
    assert result["mape"] == pytest.approx(7.5)


def test_mape_all_rows_excluded_returns_nan_not_crash():
    y_true = [0, 0, 0]
    y_pred = [10, 20, 30]
    result = compute_metrics(y_true, y_pred)
    assert math.isnan(result["mape"])
    # other metrics must still be computed fine
    assert result["mae"] == pytest.approx(20.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([1, 2, 3], [1, 2])


def test_empty_arrays_raise():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_result_keys_present():
    result = compute_metrics([100, 200], [110, 190])
    expected_keys = {
        "mae", "rmse", "r2", "median_abs_error", "mape", "bias",
        "pct_within_500", "pct_within_1000", "pct_within_2500", "pct_within_5000",
        "error_distribution", "n",
    }
    assert expected_keys <= set(result.keys())
    assert set(result["error_distribution"].keys()) == {"p10", "p25", "p50", "p75", "p90"}
