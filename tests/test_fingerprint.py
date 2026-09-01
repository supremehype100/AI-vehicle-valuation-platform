"""Tests for training/fingerprint.py — dataset/version identifiers stored
alongside every model_artifacts row."""
from training.fingerprint import compute_dataset_fingerprint, compute_proxy_fingerprint


def test_dataset_fingerprint_is_order_independent():
    a = compute_dataset_fingerprint([3, 1, 2])
    b = compute_dataset_fingerprint([1, 2, 3])
    assert a == b


def test_dataset_fingerprint_differs_for_different_ids():
    a = compute_dataset_fingerprint([1, 2, 3])
    b = compute_dataset_fingerprint([1, 2, 4])
    assert a != b


def test_dataset_fingerprint_differs_for_different_counts():
    a = compute_dataset_fingerprint([1, 2, 3])
    b = compute_dataset_fingerprint([1, 2, 3, 4])
    assert a != b


def test_dataset_fingerprint_is_deterministic_across_calls():
    ids = [42, 17, 99, 3]
    assert compute_dataset_fingerprint(ids) == compute_dataset_fingerprint(list(reversed(ids)))


def test_proxy_fingerprint_is_deterministic():
    a = compute_proxy_fingerprint(100, 20, ["make", "year"], "vehicles.csv")
    b = compute_proxy_fingerprint(100, 20, ["make", "year"], "vehicles.csv")
    assert a == b


def test_proxy_fingerprint_differs_when_inputs_differ():
    a = compute_proxy_fingerprint(100, 20, ["make", "year"], "vehicles.csv")
    b = compute_proxy_fingerprint(101, 20, ["make", "year"], "vehicles.csv")
    assert a != b
