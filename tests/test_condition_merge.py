"""
tests/test_condition_merge.py

Proves the precedence rule in condition/merge.py is correctly enforced,
in particular the safety property that matters most for this pipeline:
low-precedence AI-extracted data can NEVER overwrite a manually-validated
field, no matter how confident the model claims to be.
"""
from __future__ import annotations

from condition.merge import merge_condition_fields, derive_condition_values
from config import (
    PRECEDENCE_MANUALLY_VALIDATED,
    PRECEDENCE_TRUSTED_STRUCTURED,
    PRECEDENCE_AI_EXTRACTED,
    PRECEDENCE_INFERRED_WEAK,
)


def test_ai_cannot_overwrite_manually_validated_field():
    """The central safety property: precedence 20 (AI) must never beat
    precedence 40 (manually validated), even across many repeated attempts
    and even when the AI claims high confidence."""
    current_values = {"condition_overall": "excellent"}
    current_provenance = {
        "condition_overall": {
            "source": "manual",
            "precedence": PRECEDENCE_MANUALLY_VALIDATED,
            "model_version": None,
            "confidence": 1.0,
            "set_at": "2026-01-01T00:00:00+00:00",
            "manually_validated": True,
        }
    }

    new_values = {"condition_overall": "poor"}

    merged_values, merged_provenance = merge_condition_fields(
        current_values=current_values,
        current_provenance=current_provenance,
        new_values=new_values,
        new_source="ai",
        new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b",
        confidence=0.99,
    )

    assert merged_values["condition_overall"] == "excellent", (
        "AI-extracted value must not overwrite a manually-validated field"
    )
    assert merged_provenance["condition_overall"]["source"] == "manual"
    assert merged_provenance["condition_overall"]["precedence"] == PRECEDENCE_MANUALLY_VALIDATED
    assert merged_provenance["condition_overall"]["manually_validated"] is True

    # Original inputs must not be mutated (pure function contract).
    assert current_values["condition_overall"] == "excellent"
    assert current_provenance["condition_overall"]["source"] == "manual"


def test_ai_cannot_overwrite_trusted_structured_field():
    """Same property one tier down: precedence 20 (AI) must not beat
    precedence 30 (trusted structured data, e.g. a dealer-provided spec)."""
    current_values = {"condition_accident_indicated": False}
    current_provenance = {
        "condition_accident_indicated": {
            "source": "structured",
            "precedence": PRECEDENCE_TRUSTED_STRUCTURED,
            "model_version": None,
            "confidence": None,
            "set_at": "2026-01-01T00:00:00+00:00",
            "manually_validated": False,
        }
    }
    new_values = {"condition_accident_indicated": True}

    merged_values, merged_provenance = merge_condition_fields(
        current_values, current_provenance, new_values,
        new_source="ai", new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b", confidence=0.8,
    )

    assert merged_values["condition_accident_indicated"] is False
    assert merged_provenance["condition_accident_indicated"]["source"] == "structured"


def test_ai_can_refresh_stale_ai_value_equal_precedence():
    """Re-running the pipeline (e.g. after a prompt/model tweak) must be able
    to refresh a field it previously set itself — equal precedence wins per
    the non-strict >= rule, so backfill re-runs actually do something."""
    current_values = {"condition_exterior": "fair"}
    current_provenance = {
        "condition_exterior": {
            "source": "ai",
            "precedence": PRECEDENCE_AI_EXTRACTED,
            "model_version": "ollama:qwen2.5vl:7b",
            "confidence": 0.4,
            "set_at": "2026-01-01T00:00:00+00:00",
            "manually_validated": False,
        }
    }
    new_values = {"condition_exterior": "good"}

    merged_values, merged_provenance = merge_condition_fields(
        current_values, current_provenance, new_values,
        new_source="ai", new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b", confidence=0.9,
    )

    assert merged_values["condition_exterior"] == "good"
    assert merged_provenance["condition_exterior"]["confidence"] == 0.9


def test_ai_can_overwrite_lower_precedence_inferred_value():
    current_values = {"condition_mechanical": "unknown"}
    current_provenance = {
        "condition_mechanical": {
            "source": "inferred",
            "precedence": PRECEDENCE_INFERRED_WEAK,
            "model_version": None,
            "confidence": 0.1,
            "set_at": "2026-01-01T00:00:00+00:00",
            "manually_validated": False,
        }
    }
    new_values = {"condition_mechanical": "good"}

    merged_values, merged_provenance = merge_condition_fields(
        current_values, current_provenance, new_values,
        new_source="ai", new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b", confidence=0.7,
    )

    assert merged_values["condition_mechanical"] == "good"
    assert merged_provenance["condition_mechanical"]["source"] == "ai"


def test_brand_new_field_with_no_prior_provenance_always_gets_set():
    """A field never seen before (no provenance entry at all) must always be
    set regardless of precedence level — default-missing precedence is -1,
    so even the lowest tier (PRECEDENCE_INFERRED_WEAK=10) wins."""
    current_values: dict = {}
    current_provenance: dict = {}
    new_values = {"condition_interior": "good"}

    merged_values, merged_provenance = merge_condition_fields(
        current_values, current_provenance, new_values,
        new_source="inferred", new_precedence=PRECEDENCE_INFERRED_WEAK,
        model_version=None, confidence=None,
    )

    assert merged_values["condition_interior"] == "good"
    assert merged_provenance["condition_interior"]["precedence"] == PRECEDENCE_INFERRED_WEAK

    # Also true at AI precedence.
    merged_values2, merged_provenance2 = merge_condition_fields(
        {}, {}, {"condition_interior": "excellent"},
        new_source="ai", new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b", confidence=0.5,
    )
    assert merged_values2["condition_interior"] == "excellent"


def test_fields_absent_from_new_values_are_left_untouched():
    current_values = {"condition_overall": "good", "condition_exterior": "fair"}
    current_provenance = {
        "condition_overall": {"source": "manual", "precedence": PRECEDENCE_MANUALLY_VALIDATED,
                               "model_version": None, "confidence": None, "set_at": "x", "manually_validated": True},
    }
    merged_values, merged_provenance = merge_condition_fields(
        current_values, current_provenance, new_values={"condition_interior": "good"},
        new_source="ai", new_precedence=PRECEDENCE_AI_EXTRACTED,
        model_version="ollama:qwen2.5vl:7b", confidence=0.5,
    )
    assert merged_values["condition_overall"] == "good"
    assert merged_values["condition_exterior"] == "fair"
    assert merged_values["condition_interior"] == "good"


def test_derive_condition_values_maps_direct_fields():
    raw = {
        "overall_condition": "good",
        "exterior_condition": "fair",
        "interior_condition": "good",
        "mechanical_condition": "unknown",
        "accident_indicated": False,
        "rust_or_corrosion": None,
        "water_or_flood_indicators": False,
        "smoke_or_pet_indicators": None,
        "signs_of_neglect": False,
        "signs_of_recent_repair": True,
        "positive_indicators": ["new tires", "single owner"],
        "undisclosed_issues": [],
        "confidence": 0.75,
        "evidence": "some evidence",
    }
    values = derive_condition_values(raw)
    assert values["condition_overall"] == "good"
    assert values["condition_accident_indicated"] is False
    assert values["condition_positive_notes"] == "new tires; single owner"
    assert values["condition_undisclosed_issue_flag"] is False


def test_derive_condition_values_omits_empty_positive_notes_but_sets_undisclosed_flag():
    raw = {
        "overall_condition": "poor",
        "positive_indicators": [],
        "undisclosed_issues": ["rust under rear bumper not mentioned in listing"],
        "confidence": 0.6,
    }
    values = derive_condition_values(raw)
    assert "condition_positive_notes" not in values
    assert values["condition_undisclosed_issue_flag"] is True
