from __future__ import annotations

"""
training/splits.py

Leakage-aware train/test splitting for listing data.

WHY A RANDOM TRAIN/TEST SPLIT IS THE WRONG TOOL HERE
------------------------------------------------------
A plain `sklearn.model_selection.train_test_split` (random row shuffle) is
appropriate when rows are independent samples from a stationary distribution.
Vehicle listing data violates that assumption in several concrete ways:

  1. Reposts / relists. The same physical vehicle is frequently re-posted
     (price drop, listing expired and relisted, cross-posted to multiple
     feeds) producing multiple near-duplicate rows. `Listing.fingerprint`
     is a hash of (year, make, model, mileage-bucket, city) designed to
     catch this. A random split will happily put two reposts of the SAME
     car — often with nearly identical or literally identical price/mileage
     — into both train and test, letting the model "memorize" that exact
     vehicle instead of learning to generalize. The reported test-set error
     is then optimistic (leaked).

  2. Near-duplicate listings. Even without a shared fingerprint, near-twin
     listings (same trim/mileage-bucket, different sellers) are common in a
     scraped marketplace and share almost all of their feature values —
     splitting them across train/test has the same memorization effect as
     #1, just softer.

  3. VIN identity leakage. `Listing.vin` is the strongest possible identity
     signal — if a VIN shows up in both train and test (e.g. the vehicle
     was scraped once, sold, and later reappeared from a different seller
     or source), the model can effectively "look up" that VIN's price
     rather than predict it from general features. This is the single
     hardest leak to justify ignoring, so it gets its own enforcement step
     (`enforce_vin_disjoint` / `assert_no_vin_leakage`) run right before
     training, not just at split time.

  4. Price drift over time. Used-car prices are not stationary — they move
     with the market (seasonality, macro conditions, model-year turnover).
     A model whose test set is drawn uniformly at random from the same time
     range as its training set gets evaluated on an easier, IID-in-time
     task than the one it actually faces in production: predicting the
     price of a car being listed *now*, having only ever seen *past* data.
     A time-aware split (train = older listings, test = most recent slice)
     is the only split that matches how the model is actually used, and
     is the split every pipeline in this package uses instead of a random
     holdout.

Pipeline order used by every *_traintest.py caller:

    dedupe_by_fingerprint(listings)
        -> time_aware_split(listings, test_fraction)
        -> enforce_vin_disjoint(train, test)
        -> assert_no_vin_leakage(train, test)   # safety check right before training
"""

from datetime import datetime, timezone


def _fingerprint(listing) -> str | None:
    return getattr(listing, "fingerprint", None)


def _vin(listing) -> str | None:
    return getattr(listing, "vin", None)


def _to_epoch(dt: datetime | None) -> float | None:
    """Normalize a (possibly tz-aware, possibly naive) datetime to a
    comparable epoch float. Naive datetimes are assumed UTC (matches how
    this project stores timestamps). Returns None for None input."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.timestamp()
    return dt.replace(tzinfo=timezone.utc).timestamp()


def dedupe_by_fingerprint(listings):
    """
    Keep only the earliest (by `first_seen`) row per non-null fingerprint.
    All rows.

    Rows with a NULL fingerprint cannot be deduped (we have no identity
    signal to group them on) — they are kept as-is rather than dropped.
    This is a deliberate choice: silently dropping unfingerprinted rows
    would bias the dataset (e.g. toward listings from sources that happen
    to populate fingerprint) and lose real signal for no leakage benefit,
    since a null fingerprint can't cause a duplicate-identity split
    problem in the first place (there's nothing to match it against).

    Ties (identical or missing first_seen among duplicates of the same
    fingerprint) are broken by keeping the first one encountered in
    iteration order, for determinism.

    Preserves the original relative ordering of the input.
    """
    listings = list(listings)
    INF = float("inf")

    best_for_fp: dict[str, tuple[float, int]] = {}  # fingerprint -> (sort_key, index)
    for idx, listing in enumerate(listings):
        fp = _fingerprint(listing)
        if fp is None:
            continue
        key = _to_epoch(getattr(listing, "first_seen", None))
        if key is None:
            key = INF  # unknown first_seen: least preferred, but still a valid candidate
        current = best_for_fp.get(fp)
        if current is None or key < current[0]:
            best_for_fp[fp] = (key, idx)

    keep_indices = {idx for _, idx in best_for_fp.values()}

    result = []
    for idx, listing in enumerate(listings):
        fp = _fingerprint(listing)
        if fp is None or idx in keep_indices:
            result.append(listing)
    return result


def time_aware_split(listings, test_fraction: float = 0.15):
    """
    Sort listings by `first_seen` (falling back to `posted_at` when
    `first_seen` is missing), then take the most-recent `test_fraction`
    of rows as the test set and the rest as train.

    Rows missing BOTH `first_seen` and `posted_at` are treated as oldest
    (sorted first / into train) — we'd rather under-represent
    unknown-time rows in the "future" test set than risk them
    masquerading as recent data the model is evaluated against.

    Returns (train_listings, test_listings), both plain lists.
    """
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")

    listings = list(listings)

    def sort_key(listing):
        dt = getattr(listing, "first_seen", None)
        if dt is None:
            dt = getattr(listing, "posted_at", None)
        epoch = _to_epoch(dt)
        return epoch if epoch is not None else float("-inf")

    ordered = sorted(listings, key=sort_key)
    n = len(ordered)
    n_test = min(max(int(round(n * test_fraction)), 0), n)

    if n_test == 0:
        return ordered, []

    split_idx = n - n_test
    return ordered[:split_idx], ordered[split_idx:]


def enforce_vin_disjoint(train_listings, test_listings):
    """
    If a VIN appears in both train and test, move ALL rows sharing that VIN
    into train (the earlier/safer side — it's fine for the model to train
    on a VIN, never fine for it to be tested on one it could have seen).

    Returns (train_listings, test_listings), both plain lists.
    """
    train_listings = list(train_listings)
    test_listings = list(test_listings)

    train_vins = {_vin(l) for l in train_listings if _vin(l)}
    test_vins = {_vin(l) for l in test_listings if _vin(l)}
    overlap = train_vins & test_vins

    if not overlap:
        return train_listings, test_listings

    new_train = list(train_listings)
    new_test = []
    for listing in test_listings:
        if _vin(listing) in overlap:
            new_train.append(listing)
        else:
            new_test.append(listing)

    return new_train, new_test


def assert_no_vin_leakage(train_listings, test_listings) -> None:
    """
    Safety check meant to run immediately before training. Raises
    AssertionError with a clear message if any non-null VIN appears in
    both train and test sets.
    """
    train_vins = {_vin(l) for l in train_listings if _vin(l)}
    test_vins = {_vin(l) for l in test_listings if _vin(l)}
    overlap = train_vins & test_vins

    if overlap:
        sample = sorted(overlap)[:10]
        more = "..." if len(overlap) > 10 else ""
        raise AssertionError(
            f"VIN leakage detected: {len(overlap)} VIN(s) appear in both train "
            f"and test sets: {sample}{more}. Run enforce_vin_disjoint() before "
            f"training."
        )
