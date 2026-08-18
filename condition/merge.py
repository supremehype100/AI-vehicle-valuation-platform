"""
condition/merge.py

Pure, DB-free logic for merging a new AI condition extraction into a
listing's current "best known" condition fields, respecting a per-field
precedence ordering so that low-confidence AI output can never clobber
trusted/manually-validated data. No network or database access happens in
this module — that's what makes it independently testable and safe to call
from anywhere (backfill jobs, admin tools, future manual-review UI) without
worrying about side effects.

Field mapping — ConditionAssessmentResult (condition/schema.py) -> Listing
column (models.py):

    overall_condition          -> condition_overall
    exterior_condition         -> condition_exterior
    interior_condition         -> condition_interior
    mechanical_condition       -> condition_mechanical
    accident_indicated         -> condition_accident_indicated
    rust_or_corrosion          -> condition_rust_indicated
    water_or_flood_indicators  -> condition_flood_indicated
    smoke_or_pet_indicators    -> condition_smoke_pet_indicated
    signs_of_neglect           -> condition_neglect_indicated
    signs_of_recent_repair     -> condition_recent_repair_indicated

Derived (list/notes-shaped) fields, folded into the two remaining Listing
columns since there is no dedicated column per list field:

    positive_indicators (list[str]) -> condition_positive_notes (str)
        Joined with "; " into a short human-readable string. Only produced
        when the list is non-empty — an empty list means "the model didn't
        specifically call anything out as positive this run", not "there is
        nothing positive", so we deliberately do NOT overwrite a previously
        recorded note with nothing on a re-run.
    undisclosed_issues (list[str])  -> condition_undisclosed_issue_flag (bool)
        Always set to bool(list) — "no undisclosed issues found" is a real,
        useful answer (distinct from "unknown"), so this one IS always
        produced, even when the list is empty.

Fields present in ConditionAssessmentResult but with no corresponding
Listing column (tire_condition, glass_and_lights_condition, warning_lights,
modifications, missing_or_damaged_components, paint_or_body_issues,
paint_or_body_notes) are intentionally NOT projected onto Listing — they
remain fully available in ConditionAssessment.raw_extraction (the audit
trail) for now. Adding dedicated columns for any of these is a model
migration, out of scope here.

Precedence rule (config.PRECEDENCE_*): higher number wins. A field is
overwritten if and only if `new_precedence >= current_precedence`, where
`current_precedence` is read from the per-field provenance map (defaulting
to -1, i.e. "no prior data", if the field has never been set). This is a
non-strict inequality deliberately: re-running the SAME tier of pipeline
(e.g. AI extraction again after a model/prompt tweak) is allowed to refresh
its own prior output, but nothing below a tier can ever cross above it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import PRECEDENCE_MANUALLY_VALIDATED

# ConditionAssessmentResult field -> Listing column, for the fields that map
# 1:1 with no transformation.
DIRECT_FIELD_MAP: dict[str, str] = {
    "overall_condition": "condition_overall",
    "exterior_condition": "condition_exterior",
    "interior_condition": "condition_interior",
    "mechanical_condition": "condition_mechanical",
    "accident_indicated": "condition_accident_indicated",
    "rust_or_corrosion": "condition_rust_indicated",
    "water_or_flood_indicators": "condition_flood_indicated",
    "smoke_or_pet_indicators": "condition_smoke_pet_indicated",
    "signs_of_neglect": "condition_neglect_indicated",
    "signs_of_recent_repair": "condition_recent_repair_indicated",
}

# Full set of Listing columns the merge pipeline ever reads/writes — the
# direct-mapped columns above plus the two derived ones. Exposed so callers
# (backfill/backfill_conditions.py) can build a generic `current_values`
# dict from a Listing (or listing-like/mock object) without hardcoding the
# column list a second time.
LISTING_CONDITION_COLUMNS: tuple[str, ...] = tuple(
    sorted(set(DIRECT_FIELD_MAP.values()) | {"condition_positive_notes", "condition_undisclosed_issue_flag"})
)


def derive_condition_values(raw_extraction: dict) -> dict[str, Any]:
    """
    Turns a ConditionAssessmentResult (as a plain dict, e.g. via
    `.model_dump()`) into the dict of `Listing.condition_*` column values a
    successful extraction proposes to set. This is the `new_values` input to
    `merge_condition_fields` below.

    Only includes keys that this extraction actually has an opinion on — see
    module docstring for why `condition_positive_notes` is omitted (not set
    to empty) when there are no positive indicators this run, while
    `condition_undisclosed_issue_flag` is always included.
    """
    values: dict[str, Any] = {}

    for src_field, dest_column in DIRECT_FIELD_MAP.items():
        if src_field in raw_extraction:
            values[dest_column] = raw_extraction[src_field]

    positive_indicators = raw_extraction.get("positive_indicators") or []
    if positive_indicators:
        values["condition_positive_notes"] = "; ".join(str(item) for item in positive_indicators)

    # Always a real answer ("no undisclosed issues found this run" is
    # meaningful), unlike the other derived field above.
    undisclosed_issues = raw_extraction.get("undisclosed_issues") or []
    values["condition_undisclosed_issue_flag"] = bool(undisclosed_issues)

    return values


def merge_condition_fields(
    current_values: dict[str, Any],
    current_provenance: dict[str, Any],
    new_values: dict[str, Any],
    new_source: str,
    new_precedence: int,
    model_version: str | None,
    confidence: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Pure merge: no DB, no network, no mutation of the inputs. Returns
    (merged_values, merged_provenance) as NEW dicts.

    Rule: for each field in `new_values`, only overwrite
    `current_values[field]` if
        new_precedence >= current_provenance.get(field, {}).get("precedence", -1)
    Fields not present in `new_values` are left completely untouched (their
    current value AND provenance are carried over unchanged).

    `current_provenance` shape (per field):
        {"source": str, "precedence": int, "model_version": str|None,
         "confidence": float|None, "set_at": iso8601 str,
         "manually_validated": bool}
    """
    merged_values = dict(current_values or {})
    merged_provenance = {
        field: dict(entry) for field, entry in (current_provenance or {}).items()
    }

    set_at = datetime.now(timezone.utc).isoformat()

    for field, new_value in (new_values or {}).items():
        existing_entry = merged_provenance.get(field)
        existing_precedence = existing_entry.get("precedence", -1) if existing_entry else -1

        if new_precedence >= existing_precedence:
            merged_values[field] = new_value
            merged_provenance[field] = {
                "source": new_source,
                "precedence": new_precedence,
                "model_version": model_version,
                "confidence": confidence,
                "set_at": set_at,
                "manually_validated": new_precedence >= PRECEDENCE_MANUALLY_VALIDATED,
            }
        # else: new data is lower-precedence than what's already trusted for
        # this field — leave both value and provenance exactly as they were.

    return merged_values, merged_provenance
