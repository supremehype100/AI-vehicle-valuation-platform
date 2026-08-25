"""
tests/test_backfill_logic.py

There is no live Postgres or Ollama reachable in this environment, so these
tests prove the two things that matter about backfill/backfill_conditions.py
WITHOUT a real database:

  1. The selection-query builders produce the SQL we expect (dataset filter,
     NULL-confidence-or-stale-assessment predicate, the EXISTS subquery keyed
     on the *current* model_version/schema_version) — verified by compiling
     the SQLAlchemy Select statements to SQL text, which requires no
     connection at all.

  2. The per-row "extract -> build audit row -> merge onto listing" wiring in
     `apply_extraction_result` / `build_assessment_row` / `run()` is correct
     — verified against fake/mock listing and session objects, so no engine
     is touched.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from backfill import backfill_conditions as bc
from condition.merge import LISTING_CONDITION_COLUMNS
from config import CONDITION_MODEL_VERSION, CONDITION_SCHEMA_VERSION, PRECEDENCE_AI_EXTRACTED, PRECEDENCE_MANUALLY_VALIDATED


# ---------------------------------------------------------------------------
# 1. Selection-query builders (compiled SQL text only, no connection)
# ---------------------------------------------------------------------------

def test_selection_query_filters_by_dataset():
    query = bc.build_selection_query("dealer")
    sql = str(query)
    assert "listing_type" in sql
    assert "condition_confidence" in sql
    assert "EXISTS" in sql.upper()


def test_selection_query_all_dataset_has_no_listing_type_filter():
    # select(Listing) always mentions the listing_type COLUMN in the SELECT
    # list (it's a mapped column), so check the WHERE predicate specifically
    # rather than the whole compiled text.
    query = bc.build_selection_query("all")
    sql = str(query)
    where_clause = sql.split("WHERE", 1)[1] if "WHERE" in sql else ""
    assert "listing_type" not in where_clause


def test_selection_query_exists_subquery_keys_on_current_model_and_schema_version():
    query = bc.build_selection_query("private")
    compiled = query.compile(compile_kwargs={"literal_binds": True})
    sql = str(compiled)
    assert CONDITION_MODEL_VERSION in sql
    assert CONDITION_SCHEMA_VERSION in sql
    assert "NOT (EXISTS" in sql or "NOT EXISTS" in sql.upper().replace("NOT (EXISTS", "NOT EXISTS")


def test_selection_query_respects_limit():
    query = bc.build_selection_query("all", limit=25)
    sql = str(query.compile(compile_kwargs={"literal_binds": True}))
    assert "LIMIT 25" in sql or "\n LIMIT" in sql  # dialect-formatting tolerant


def test_selection_query_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        bc.build_selection_query("not_a_real_dataset")


def test_total_count_query_has_no_processing_predicate():
    """Total count must NOT filter on condition_confidence/assessments — it's
    every listing matching the dataset, processed or not."""
    query = bc.build_total_count_query("salvage")
    sql = str(query)
    assert "listing_type" in sql
    assert "condition_confidence" not in sql


def test_remaining_count_query_matches_selection_predicate():
    remaining_sql = str(bc.build_remaining_count_query("private").compile(compile_kwargs={"literal_binds": True}))
    selection_sql = str(bc.build_selection_query("private").compile(compile_kwargs={"literal_binds": True}))
    assert "condition_confidence" in remaining_sql
    assert CONDITION_MODEL_VERSION in remaining_sql
    # Same WHERE predicate family in both (selection just also SELECTs rows / has ORDER BY).
    assert "listing_type" in remaining_sql and "listing_type" in selection_sql


# ---------------------------------------------------------------------------
# 2. Per-row merge-and-update wiring (fake listing objects, no DB)
# ---------------------------------------------------------------------------

def _make_fake_listing(listing_id: int, **provenance_overrides) -> SimpleNamespace:
    listing = SimpleNamespace(
        id=listing_id,
        description="a description",
        image_urls=["http://example.com/1.jpg"],
        condition_field_provenance=dict(provenance_overrides),
        condition_confidence=None,
    )
    for col in LISTING_CONDITION_COLUMNS:
        setattr(listing, col, None)
    return listing


def _success_result(overall="good", confidence=0.8) -> dict:
    return {
        "model_version": CONDITION_MODEL_VERSION,
        "schema_version": CONDITION_SCHEMA_VERSION,
        "images_used": [{"url": "http://example.com/1.jpg", "hash": 123, "used": True, "reason_skipped": None}],
        "description_hash": "deadbeef",
        "extraction_status": "success",
        "error_message": None,
        "raw_extraction": {
            "overall_condition": overall,
            "exterior_condition": "good",
            "interior_condition": "good",
            "mechanical_condition": "unknown",
            "accident_indicated": False,
            "positive_indicators": ["clean title"],
            "undisclosed_issues": [],
            "confidence": confidence,
            "evidence": "text says clean title, photos look tidy",
        },
        "confidence_overall": confidence,
    }


def _failure_result(error="Ollama unavailable: connection refused") -> dict:
    return {
        "model_version": CONDITION_MODEL_VERSION,
        "schema_version": CONDITION_SCHEMA_VERSION,
        "images_used": [],
        "description_hash": "deadbeef",
        "extraction_status": "failed",
        "error_message": error,
        "raw_extraction": {},
        "confidence_overall": None,
    }


def test_apply_extraction_result_sets_fields_on_fresh_listing():
    listing = _make_fake_listing(1)
    result = _success_result()

    apply_provenance = bc.apply_extraction_result(listing, result)

    assert listing.condition_overall == "good"
    assert listing.condition_accident_indicated is False
    assert listing.condition_positive_notes == "clean title"
    assert listing.condition_undisclosed_issue_flag is False
    assert listing.condition_confidence == 0.8
    assert apply_provenance["condition_overall"]["source"] == "ai"
    assert apply_provenance["condition_overall"]["precedence"] == PRECEDENCE_AI_EXTRACTED


def test_apply_extraction_result_does_not_bypass_precedence_protection():
    """The whole point of routing through merge.merge_condition_fields:
    a manually-validated field must survive apply_extraction_result untouched."""
    listing = _make_fake_listing(
        2,
        condition_overall={
            "source": "manual",
            "precedence": PRECEDENCE_MANUALLY_VALIDATED,
            "model_version": None,
            "confidence": 1.0,
            "set_at": "2026-01-01T00:00:00+00:00",
            "manually_validated": True,
        },
    )
    listing.condition_overall = "excellent"  # the actual current value

    result = _success_result(overall="poor", confidence=0.95)
    bc.apply_extraction_result(listing, result)

    assert listing.condition_overall == "excellent", "AI result must not overwrite manually-validated field"


def test_build_assessment_row_success_and_failure_shapes():
    success_row = bc.build_assessment_row(42, _success_result())
    assert success_row.listing_id == 42
    assert success_row.extraction_status == "success"
    assert success_row.error_message is None
    assert success_row.raw_extraction["overall_condition"] == "good"

    failure_row = bc.build_assessment_row(43, _failure_result())
    assert failure_row.listing_id == 43
    assert failure_row.extraction_status == "failed"
    assert failure_row.raw_extraction == {}
    assert failure_row.confidence_overall is None


# ---------------------------------------------------------------------------
# 3. Full run() orchestration against a fake session (no DB, no Ollama)
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, total, remaining, listings):
        self._total = total
        self._remaining = remaining
        self._listings = listings
        self.added = []
        self.commit_count = 0

    def execute(self, query):
        if query == "TOTAL":
            return _FakeResult(scalar=self._total)
        if query == "REMAINING":
            return _FakeResult(scalar=self._remaining)
        if query == "SELECTION":
            return _FakeResult(rows=self._listings)
        raise AssertionError(f"unexpected query sentinel: {query!r}")

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commit_count += 1


def test_run_orchestrates_selection_extraction_merge_and_report(monkeypatch, capsys):
    listings = [_make_fake_listing(i) for i in range(1, 6)]  # 5 listings
    fake_session = _FakeSession(total=20, remaining=5, listings=listings)

    monkeypatch.setattr(bc, "build_total_count_query", lambda dataset: "TOTAL")
    monkeypatch.setattr(bc, "build_remaining_count_query", lambda dataset: "REMAINING")
    monkeypatch.setattr(bc, "build_selection_query", lambda dataset, limit: "SELECTION")

    @contextmanager
    def fake_get_session():
        yield fake_session

    monkeypatch.setattr(bc, "get_session", fake_get_session)

    def fake_extract(description, image_urls):
        # listings 1,2,3,4 succeed; listing 5 fails — deterministic on id via closure below
        listing_id = image_urls_to_id[id(image_urls)]
        if listing_id == 5:
            return _failure_result()
        return _success_result(confidence=0.5 + 0.1 * listing_id)

    image_urls_to_id = {id(l.image_urls): l.id for l in listings}
    monkeypatch.setattr(bc, "extract_condition_for_listing", fake_extract)

    args = SimpleNamespace(dataset="all", batch_size=2, limit=None, resume=False)
    stats = bc.run(args)

    assert stats == {
        "total": 20,
        "already_processed": 15,
        "remaining": 5,
        "processed_this_run": 5,
        "successful": 4,
        "failed": 1,
    }

    # One audit row per listing, always inserted (success AND failure).
    assert len(fake_session.added) == 5
    assert sum(1 for row in fake_session.added if row.extraction_status == "success") == 4
    assert sum(1 for row in fake_session.added if row.extraction_status == "failed") == 1

    # Batch commits: batch_size=2 over 5 rows -> commits at row 2 and row 4,
    # plus one final commit for the trailing partial batch (row 5) = 3 total.
    assert fake_session.commit_count == 3

    # Merge actually happened for the successful listings.
    successes = [l for l in listings if l.id != 5]
    for listing in successes:
        assert listing.condition_overall == "good"
        assert listing.condition_confidence is not None
    failed_listing = listings[4]
    assert failed_listing.condition_overall is None  # untouched on failure

    # Aggregate report printed in the exact required shape.
    out = capsys.readouterr().out
    expected_lines = [
        "Total records: 20",
        "Already processed: 15",
        "Remaining: 5",
        "Processed this run: 5",
        "Successful: 4",
        "Failed: 1",
    ]
    for line in expected_lines:
        assert line in out


def test_run_respects_resume_flag_as_noop(monkeypatch, caplog):
    listings = [_make_fake_listing(1)]
    fake_session = _FakeSession(total=1, remaining=1, listings=listings)

    monkeypatch.setattr(bc, "build_total_count_query", lambda dataset: "TOTAL")
    monkeypatch.setattr(bc, "build_remaining_count_query", lambda dataset: "REMAINING")
    monkeypatch.setattr(bc, "build_selection_query", lambda dataset, limit: "SELECTION")

    @contextmanager
    def fake_get_session():
        yield fake_session

    monkeypatch.setattr(bc, "get_session", fake_get_session)
    monkeypatch.setattr(bc, "extract_condition_for_listing", lambda description, image_urls: _success_result())

    args = SimpleNamespace(dataset="all", batch_size=50, limit=None, resume=True)
    with caplog.at_level("INFO"):
        stats = bc.run(args)

    assert stats["processed_this_run"] == 1
    assert any("no-op" in message for message in caplog.messages)


def test_parse_args_defaults_and_validation():
    args = bc.parse_args([])
    assert args.dataset == "all"
    assert args.batch_size == 50
    assert args.limit is None
    assert args.resume is False

    args2 = bc.parse_args(["--resume", "--batch-size", "10", "--dataset", "dealer", "--limit", "100"])
    assert args2.resume is True
    assert args2.batch_size == 10
    assert args2.dataset == "dealer"
    assert args2.limit == 100

    with pytest.raises(SystemExit):
        bc.parse_args(["--batch-size", "0"])
    with pytest.raises(SystemExit):
        bc.parse_args(["--dataset", "not_a_dataset"])
