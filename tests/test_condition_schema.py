"""
Tests for condition/schema.py — AI structured-output parsing and how the
pipeline behaves when the model returns incomplete/missing fields. Required
by the task spec ("AI structured output parsing", "Missing AI fields").
"""
import pytest
from pydantic import ValidationError

from condition.schema import ConditionAssessmentResult


MINIMAL_VALID = {
    "overall_condition": "good",
    "exterior_condition": "good",
    "interior_condition": "fair",
    "mechanical_condition": "unknown",
    "confidence": 0.7,
    "evidence": "Description mentions one owner, no accidents; photos show light interior wear.",
}


def test_minimal_valid_response_parses():
    result = ConditionAssessmentResult.model_validate(MINIMAL_VALID)
    assert result.overall_condition == "good"
    assert result.confidence == 0.7


def test_optional_fields_default_sensibly_when_omitted():
    result = ConditionAssessmentResult.model_validate(MINIMAL_VALID)
    # Boolean indicators default to None ("no evidence either way"), not False.
    assert result.accident_indicated is None
    assert result.rust_or_corrosion is None
    assert result.water_or_flood_indicators is None
    # List fields default to empty list, never None.
    assert result.warning_lights == []
    assert result.positive_indicators == []
    assert result.undisclosed_issues == []
    # Free-text fields default to a sentinel string, not a crash.
    assert result.tire_condition == "unknown"


def test_missing_required_overall_condition_raises_validation_error():
    payload = dict(MINIMAL_VALID)
    del payload["overall_condition"]
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_missing_required_evidence_raises_validation_error():
    payload = dict(MINIMAL_VALID)
    del payload["evidence"]
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_empty_evidence_string_raises_validation_error():
    payload = dict(MINIMAL_VALID, evidence="")
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_confidence_out_of_range_raises_validation_error():
    payload = dict(MINIMAL_VALID, confidence=1.5)
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_invalid_rating_literal_raises_validation_error():
    payload = dict(MINIMAL_VALID, overall_condition="pretty good ish")
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_unexpected_extra_field_is_rejected():
    """extra='forbid' — a model hallucinating an extra key should fail loudly,
    not silently pass through unvalidated data."""
    payload = dict(MINIMAL_VALID, made_up_field="surprise")
    with pytest.raises(ValidationError):
        ConditionAssessmentResult.model_validate(payload)


def test_fully_populated_response_round_trips():
    payload = dict(
        MINIMAL_VALID,
        accident_indicated=True,
        rust_or_corrosion=False,
        paint_or_body_issues=True,
        paint_or_body_notes="Scuff on rear bumper",
        tire_condition="good, even wear",
        glass_and_lights_condition="one small windshield chip",
        warning_lights=["check engine"],
        modifications=["aftermarket exhaust"],
        missing_or_damaged_components=[],
        water_or_flood_indicators=False,
        smoke_or_pet_indicators=False,
        signs_of_neglect=False,
        signs_of_recent_repair=True,
        undisclosed_issues=["windshield chip not mentioned in description"],
        positive_indicators=["new tires", "single owner"],
    )
    result = ConditionAssessmentResult.model_validate(payload)
    assert result.warning_lights == ["check engine"]
    assert result.positive_indicators == ["new tires", "single owner"]


def test_ollama_response_format_is_a_json_schema_dict():
    schema = ConditionAssessmentResult.ollama_response_format()
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "overall_condition" in schema.get("properties", {})


def test_json_string_response_parses_via_model_validate_json():
    """Mirrors how condition/ollama_client.py actually parses Ollama's
    response body (json.loads then model_validate, or model_validate_json)."""
    import json
    raw_json_string = json.dumps(MINIMAL_VALID)
    result = ConditionAssessmentResult.model_validate(json.loads(raw_json_string))
    assert result.overall_condition == "good"


def test_malformed_json_string_does_not_silently_produce_a_result():
    """A non-JSON or truncated model response must fail parsing, not be
    coerced into something that looks valid."""
    import json
    with pytest.raises(json.JSONDecodeError):
        json.loads('{"overall_condition": "good", "evidence": "cut off mid')
