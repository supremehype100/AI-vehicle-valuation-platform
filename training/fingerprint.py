"""
training/fingerprint.py

Computes the `dataset_fingerprint` stored on every `model_artifacts` row —
the "dataset/version identifier" the task spec requires per model. Lets you
tell whether two training runs that produced the same row COUNT actually
trained on the same underlying rows (e.g. after a backfill changed
condition data, or after new listings were scraped in between runs).
"""
from __future__ import annotations
import hashlib


def compute_dataset_fingerprint(listing_ids: list[int]) -> str:
    """
    Deterministic fingerprint of the exact set of listing rows used for a
    training run. Order-independent (sorted before hashing) — the same set
    of rows fingerprints identically regardless of query result ordering.
    """
    joined = ",".join(str(i) for i in sorted(listing_ids))
    return hashlib.md5(joined.encode()).hexdigest()


def compute_proxy_fingerprint(*parts) -> str:
    """
    Fallback for pipelines that don't have direct access to the row IDs used
    (Model A / private_csv.py, which calls the unmodified Model.train() —
    off-limits to change — and only gets back aggregate counts, not IDs).
    Hashes whatever identifying metadata IS available (sample counts, feature
    list, model version, csv path) so re-runs with identical inputs produce
    an identical fingerprint, and runs with different data/config don't
    collide. Weaker than compute_dataset_fingerprint (won't detect the same
    COUNT of different underlying rows), documented as such wherever used.
    """
    joined = "|".join(str(p) for p in parts)
    return hashlib.md5(joined.encode()).hexdigest()
