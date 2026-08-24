"""
backfill/backfill_conditions.py

CLI that runs AI condition extraction (condition/extract.py) over listings
in the database and merges the results onto `Listing.condition_*` via
precedence rules (condition/merge.py), inserting an audit-trail row into
`condition_assessments` (models.ConditionAssessment) for every attempt,
success or failure.

Usage:
    python -m backfill.backfill_conditions [--resume] [--batch-size 50]
        [--dataset private|dealer|salvage|all] [--limit N]

--- Why this is idempotent/resumable with NO separate checkpoint table ---

The row-selection query is:

    listings WHERE listing_type matches --dataset filter
             AND ( condition_confidence IS NULL
                   OR NOT EXISTS (
                        SELECT 1 FROM condition_assessments ca
                        WHERE ca.listing_id = listings.id
                          AND ca.model_version = CONDITION_MODEL_VERSION
                          AND ca.schema_version = CONDITION_SCHEMA_VERSION
                   ) )

A listing drops out of this selection the moment it has BOTH a non-null
`condition_confidence` AND at least one `condition_assessments` row stamped
with the *current* model_version + schema_version (config.py). Both of those
become true as a side effect of successfully processing the listing once
(see `apply_extraction_result` below, and the fact that a ConditionAssessment
row is always inserted). So:

  - Killing the process mid-run and re-running the exact same command simply
    re-issues this query, which now naturally excludes every listing that
    was already finished — no separate "processed_ids" checkpoint file/table
    is needed, and there is no risk of accidentally reprocessing (expensive,
    since each row is an LLM call) or accidentally skipping a row that
    failed before a checkpoint would have been written.
  - Failed attempts do NOT satisfy the exclusion (condition_confidence stays
    NULL on failure), so a listing that failed last run is automatically
    retried next run — "resumable" includes "retry what didn't finish", not
    just "skip what did".
  - Bumping QWEN_MODEL or CONDITION_SCHEMA_VERSION automatically makes every
    listing eligible for reprocessing again under the new model/schema,
    since the EXISTS check is keyed on both — this is also how a full
    re-backfill after a prompt/model upgrade is triggered, with no special
    "--force" flag needed.

`--resume` is accepted for CLI-compatibility/documentation purposes but is a
no-op: the behavior above is unconditional, there is nothing extra to opt
into. `--batch-size` controls how often we COMMIT (not how many rows we
select) — sized so a crash mid-run loses at most one partial batch of
already-completed work, without a commit-per-row's overhead.
"""
from __future__ import annotations

import argparse
import logging

from sqlalchemy import and_, func, not_, or_, select

from config import (
    CONDITION_MODEL_VERSION,
    CONDITION_SCHEMA_VERSION,
    PRECEDENCE_AI_EXTRACTED,
    VALID_LISTING_TYPES,
)
from condition.extract import extract_condition_for_listing
from condition.merge import LISTING_CONDITION_COLUMNS, derive_condition_values, merge_condition_fields
from models import ConditionAssessment, Listing
from Session import get_session

logger = logging.getLogger(__name__)

DATASET_CHOICES = ("private", "dealer", "salvage", "all")


# ---------------------------------------------------------------------------
# Query builders — pure w.r.t. the DB (they only build SQLAlchemy Select
# statements, never execute them), so they're unit-testable by compiling to
# SQL text without a live connection. See tests/test_backfill_logic.py.
# ---------------------------------------------------------------------------

def _dataset_conditions(dataset: str) -> list:
    if dataset == "all":
        return []
    if dataset not in VALID_LISTING_TYPES:
        raise ValueError(f"Unknown dataset filter {dataset!r}, expected one of {DATASET_CHOICES}")
    return [Listing.listing_type == dataset]


def _needs_processing_clause():
    """Listing.condition_confidence IS NULL OR no fresh ConditionAssessment exists."""
    has_current_assessment = (
        select(ConditionAssessment.id)
        .where(
            ConditionAssessment.listing_id == Listing.id,
            ConditionAssessment.model_version == CONDITION_MODEL_VERSION,
            ConditionAssessment.schema_version == CONDITION_SCHEMA_VERSION,
        )
        .exists()
    )
    return or_(Listing.condition_confidence.is_(None), not_(has_current_assessment))


def build_selection_query(dataset: str, limit: int | None = None):
    """Rows to (re)process this run, oldest-id-first, optionally capped at `limit`."""
    conditions = _dataset_conditions(dataset) + [_needs_processing_clause()]
    query = select(Listing).where(and_(*conditions)).order_by(Listing.id)
    if limit:
        query = query.limit(limit)
    return query


def build_total_count_query(dataset: str):
    """Count of all listings matching the dataset filter, regardless of processed status."""
    query = select(func.count()).select_from(Listing)
    conditions = _dataset_conditions(dataset)
    if conditions:
        query = query.where(and_(*conditions))
    return query


def build_remaining_count_query(dataset: str):
    """Count of listings matching the dataset filter that still need processing."""
    conditions = _dataset_conditions(dataset) + [_needs_processing_clause()]
    return select(func.count()).select_from(Listing).where(and_(*conditions))


# ---------------------------------------------------------------------------
# Per-row merge-and-update — operates on any listing-like object (duck typed:
# needs get/setattr for the columns in LISTING_CONDITION_COLUMNS plus
# `condition_field_provenance` and `condition_confidence`), so it's testable
# against a plain mock/fake object without a live DB. See models.Listing for
# the real target.
# ---------------------------------------------------------------------------

def apply_extraction_result(
    listing,
    extraction_result: dict,
    *,
    precedence: int = PRECEDENCE_AI_EXTRACTED,
    model_version: str = CONDITION_MODEL_VERSION,
) -> dict:
    """
    Mutates `listing` in place with the precedence-merged outcome of a
    SUCCESSFUL extraction_result (as returned by
    condition.extract.extract_condition_for_listing). Callers must only
    invoke this when extraction_result["extraction_status"] == "success".

    Never bypasses condition.merge.merge_condition_fields — that function is
    the sole enforcement point for "AI can't overwrite manually validated
    data", so this function must not set any condition_* attribute any other
    way.

    Returns the merged provenance dict that was written (handy for tests).
    """
    current_values = {col: getattr(listing, col, None) for col in LISTING_CONDITION_COLUMNS}
    current_provenance = getattr(listing, "condition_field_provenance", None) or {}
    new_values = derive_condition_values(extraction_result["raw_extraction"])

    merged_values, merged_provenance = merge_condition_fields(
        current_values=current_values,
        current_provenance=current_provenance,
        new_values=new_values,
        new_source="ai",
        new_precedence=precedence,
        model_version=model_version,
        confidence=extraction_result.get("confidence_overall"),
    )

    for column, value in merged_values.items():
        setattr(listing, column, value)
    listing.condition_field_provenance = merged_provenance
    # Overall confidence is a single scalar summary of the latest successful
    # run, not a per-field precedence-gated value (there's no dedicated
    # provenance entry "for" it) — always refresh it on success.
    listing.condition_confidence = extraction_result.get("confidence_overall")

    return merged_provenance


def build_assessment_row(listing_id: int, extraction_result: dict) -> ConditionAssessment:
    """Builds the (unsaved) ConditionAssessment audit row for one attempt, success or failure."""
    return ConditionAssessment(
        listing_id=listing_id,
        model_version=extraction_result["model_version"],
        schema_version=extraction_result["schema_version"],
        images_used=extraction_result["images_used"],
        description_hash=extraction_result["description_hash"],
        extraction_status=extraction_result["extraction_status"],
        error_message=extraction_result["error_message"],
        raw_extraction=extraction_result["raw_extraction"],
        confidence_overall=extraction_result["confidence_overall"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m backfill.backfill_conditions",
        description="Backfill AI-extracted condition assessments onto listings.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Accepted for compatibility; no-op - the selection query is always resumable, see module docstring.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N processed listings.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all", help="Restrict to a listing_type, or 'all'.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N listings this run.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def print_report(stats: dict) -> None:
    print(f"Total records: {stats['total']:,}")
    print(f"Already processed: {stats['already_processed']:,}")
    print(f"Remaining: {stats['remaining']:,}")
    print(f"Processed this run: {stats['processed_this_run']:,}")
    print(f"Successful: {stats['successful']:,}")
    print(f"Failed: {stats['failed']:,}")


def run(args: argparse.Namespace) -> dict:
    """Executes one backfill run against the real database and returns the aggregate stats dict."""
    if args.resume:
        logger.info("--resume passed: no-op, selection query is unconditionally resumable (see module docstring).")

    with get_session() as session:
        total = session.execute(build_total_count_query(args.dataset)).scalar_one()
        remaining_before = session.execute(build_remaining_count_query(args.dataset)).scalar_one()
        already_processed = total - remaining_before

        listings = session.execute(build_selection_query(args.dataset, args.limit)).scalars().all()
        logger.info("Selected %d listing(s) to process this run (dataset=%s)", len(listings), args.dataset)

        processed = 0
        successful = 0
        failed = 0

        for listing in listings:
            # NEVER log listing.description or any image bytes/base64 — only
            # ids and small status fields, so this stays diagnosable across
            # a run of thousands of rows without leaking large/sensitive payloads.
            result = extract_condition_for_listing(listing.description, listing.image_urls)
            session.add(build_assessment_row(listing.id, result))

            if result["extraction_status"] == "success":
                apply_extraction_result(listing, result)
                successful += 1
                logger.info("listing_id=%s status=success confidence=%.3f", listing.id, result["confidence_overall"] or 0.0)
            else:
                failed += 1
                logger.warning("listing_id=%s status=failed error=%s", listing.id, result["error_message"])

            processed += 1
            if processed % args.batch_size == 0:
                session.commit()
                logger.info("Committed batch: %d/%d listings processed this run", processed, len(listings))

        session.commit()

    stats = {
        "total": total,
        "already_processed": already_processed,
        "remaining": remaining_before,
        "processed_this_run": processed,
        "successful": successful,
        "failed": failed,
    }
    print_report(stats)
    return stats


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
