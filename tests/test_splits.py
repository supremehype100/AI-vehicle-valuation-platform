"""
tests/test_splits.py

Tests for training.splits — dedupe, time-aware split, VIN disjointness
enforcement, and the pre-training safety assertion. Uses plain Python
objects (dataclasses), not a real ORM session, per training.splits'
documented contract of working against anything with the right attributes.
No DB, no network.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from training.splits import (
    dedupe_by_fingerprint,
    time_aware_split,
    enforce_vin_disjoint,
    assert_no_vin_leakage,
)


@dataclass
class FakeListing:
    id: int
    fingerprint: str | None = None
    first_seen: datetime | None = None
    posted_at: datetime | None = None
    vin: str | None = None


BASE = datetime(2026, 1, 1)


def day(n):
    return BASE + timedelta(days=n)


# ---------------------------------------------------------------------------
# dedupe_by_fingerprint
# ---------------------------------------------------------------------------

def test_dedupe_keeps_earliest_per_fingerprint():
    listings = [
        FakeListing(id=1, fingerprint="a", first_seen=day(5)),
        FakeListing(id=2, fingerprint="a", first_seen=day(1)),  # earliest for "a"
        FakeListing(id=3, fingerprint="a", first_seen=day(3)),
        FakeListing(id=4, fingerprint="b", first_seen=day(2)),
    ]
    result = dedupe_by_fingerprint(listings)
    ids = {l.id for l in result}
    assert ids == {2, 4}


def test_dedupe_keeps_all_null_fingerprint_rows():
    listings = [
        FakeListing(id=1, fingerprint=None, first_seen=day(1)),
        FakeListing(id=2, fingerprint=None, first_seen=day(2)),
        FakeListing(id=3, fingerprint="x", first_seen=day(1)),
        FakeListing(id=4, fingerprint="x", first_seen=day(9)),
    ]
    result = dedupe_by_fingerprint(listings)
    ids = {l.id for l in result}
    # both null-fingerprint rows kept, plus only the earliest of fingerprint "x"
    assert ids == {1, 2, 3}


def test_dedupe_preserves_relative_order():
    listings = [
        FakeListing(id=1, fingerprint=None, first_seen=day(1)),
        FakeListing(id=2, fingerprint="a", first_seen=day(1)),
        FakeListing(id=3, fingerprint=None, first_seen=day(2)),
        FakeListing(id=4, fingerprint="a", first_seen=day(5)),  # dupe, dropped
    ]
    result = dedupe_by_fingerprint(listings)
    assert [l.id for l in result] == [1, 2, 3]


# ---------------------------------------------------------------------------
# time_aware_split
# ---------------------------------------------------------------------------

def test_time_aware_split_ordering_and_sizes():
    listings = [FakeListing(id=i, first_seen=day(i)) for i in range(10)]
    train, test = time_aware_split(listings, test_fraction=0.2)

    assert len(train) == 8
    assert len(test) == 2
    # test set is the most recent 2 (ids 8, 9)
    assert {l.id for l in test} == {8, 9}
    # every train first_seen predates every test first_seen
    assert max(l.first_seen for l in train) <= min(l.first_seen for l in test)


def test_time_aware_split_falls_back_to_posted_at():
    listings = [
        FakeListing(id=1, first_seen=None, posted_at=day(1)),
        FakeListing(id=2, first_seen=None, posted_at=day(5)),
        FakeListing(id=3, first_seen=day(3)),
    ]
    train, test = time_aware_split(listings, test_fraction=1 / 3)
    # most recent by (first_seen or posted_at) is id=2 (posted_at day 5)
    assert [l.id for l in test] == [2]


def test_time_aware_split_zero_fraction_returns_empty_test():
    listings = [FakeListing(id=i, first_seen=day(i)) for i in range(5)]
    train, test = time_aware_split(listings, test_fraction=0.0)
    assert len(train) == 5
    assert test == []


def test_time_aware_split_invalid_fraction_raises():
    with pytest.raises(ValueError):
        time_aware_split([], test_fraction=1.0)
    with pytest.raises(ValueError):
        time_aware_split([], test_fraction=-0.1)


# ---------------------------------------------------------------------------
# enforce_vin_disjoint
# ---------------------------------------------------------------------------

def test_enforce_vin_disjoint_moves_all_shared_vin_rows_to_train():
    train = [
        FakeListing(id=1, vin="VIN1", first_seen=day(1)),
        FakeListing(id=2, vin="VIN2", first_seen=day(2)),
    ]
    test = [
        FakeListing(id=3, vin="VIN1", first_seen=day(10)),  # repost of VIN1, would leak
        FakeListing(id=4, vin="VIN3", first_seen=day(11)),
    ]

    new_train, new_test = enforce_vin_disjoint(train, test)

    train_ids = {l.id for l in new_train}
    test_ids = {l.id for l in new_test}

    assert train_ids == {1, 2, 3}  # id=3 moved into train
    assert test_ids == {4}

    # no VIN overlap remains
    train_vins = {l.vin for l in new_train if l.vin}
    test_vins = {l.vin for l in new_test if l.vin}
    assert train_vins.isdisjoint(test_vins)


def test_enforce_vin_disjoint_noop_when_already_disjoint():
    train = [FakeListing(id=1, vin="VIN1")]
    test = [FakeListing(id=2, vin="VIN2")]
    new_train, new_test = enforce_vin_disjoint(train, test)
    assert [l.id for l in new_train] == [1]
    assert [l.id for l in new_test] == [2]


def test_enforce_vin_disjoint_ignores_null_vins():
    train = [FakeListing(id=1, vin=None)]
    test = [FakeListing(id=2, vin=None)]
    new_train, new_test = enforce_vin_disjoint(train, test)
    assert [l.id for l in new_train] == [1]
    assert [l.id for l in new_test] == [2]


# ---------------------------------------------------------------------------
# assert_no_vin_leakage
# ---------------------------------------------------------------------------

def test_assert_no_vin_leakage_raises_on_overlap():
    train = [FakeListing(id=1, vin="VIN1")]
    test = [FakeListing(id=2, vin="VIN1")]
    with pytest.raises(AssertionError, match="VIN leakage"):
        assert_no_vin_leakage(train, test)


def test_assert_no_vin_leakage_passes_when_disjoint():
    train = [FakeListing(id=1, vin="VIN1")]
    test = [FakeListing(id=2, vin="VIN2")]
    # should not raise
    assert_no_vin_leakage(train, test)


def test_assert_no_vin_leakage_passes_with_null_vins():
    train = [FakeListing(id=1, vin=None)]
    test = [FakeListing(id=2, vin=None)]
    assert_no_vin_leakage(train, test)


# ---------------------------------------------------------------------------
# Integration: dedupe -> split -> enforce -> assert, no leakage survives
# ---------------------------------------------------------------------------

def test_full_pipeline_produces_no_vin_leakage():
    listings = [
        FakeListing(id=1, fingerprint="a", vin="VIN1", first_seen=day(1)),
        FakeListing(id=2, fingerprint="a", vin="VIN1", first_seen=day(2)),  # repost, same fp+vin
        FakeListing(id=3, fingerprint="b", vin="VIN1", first_seen=day(20)),  # different fp, same VIN, recent
        FakeListing(id=4, fingerprint="c", vin="VIN2", first_seen=day(15)),
        FakeListing(id=5, fingerprint="d", vin="VIN3", first_seen=day(16)),
    ]
    deduped = dedupe_by_fingerprint(listings)
    train, test = time_aware_split(deduped, test_fraction=0.4)
    train, test = enforce_vin_disjoint(train, test)
    # must not raise
    assert_no_vin_leakage(train, test)
