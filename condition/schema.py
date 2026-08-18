"""
condition/schema.py

The structured-output contract Qwen (via Ollama) must fill in when assessing
a listing's condition from its description text + photos. This is passed
verbatim as the `format` parameter of Ollama's /api/chat call
(ConditionAssessmentResult.model_json_schema()) so the model is constrained to
emit exactly this shape — no free-form prose to parse downstream.

Design notes:
  - Rating-style fields (overall/exterior/interior/mechanical) use a small
    closed Literal set rather than free text, because these feed directly
    into `Listing.condition_*` string columns and downstream feature
    engineering (Features.py) that will eventually bucket/one-hot them —
    unconstrained free text would reintroduce the same "50 different spellings
    of 'good'" problem structured extraction is meant to solve.
  - The boolean "indicator" fields (accident_indicated, rust_or_corrosion,
    etc.) are `bool | None` rather than plain `bool`, deliberately mirroring
    the `Listing.condition_*_indicated` columns (models.py), which are also
    nullable. None means "the model found no evidence either way" — a
    meaningfully different state from False ("model saw evidence of NO
    accident"), and collapsing that distinction to False would let a
    low-confidence "couldn't tell" response silently overwrite a previously
    known field on merge.
  - List fields default to an empty list (never None) so callers can always
    iterate them without a null check.
  - `evidence` is required (min length 1) — every assessment must be
    auditable back to something the model actually saw in the text/photos,
    not just an opaque label. This is what makes the pipeline debuggable
    when it inevitably gets a listing wrong.
  - No nested BaseModels — everything is flat scalars/lists of scalars. Ollama's
    structured-output support is most reliable against a flat JSON schema
    (no $defs/$ref indirection), so paint/body condition is split into a
    plain bool + plain str instead of a nested sub-object.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConditionRating = Literal["excellent", "good", "fair", "poor", "unknown"]


class ConditionAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Headline ratings ---
    overall_condition: ConditionRating = Field(
        description="Overall condition of the vehicle taking everything into account."
    )
    exterior_condition: ConditionRating = Field(
        description="Condition of the exterior body/paint as seen in photos."
    )
    interior_condition: ConditionRating = Field(
        description="Condition of the interior (seats, dash, carpet, cleanliness)."
    )
    mechanical_condition: ConditionRating = Field(
        description="Condition of mechanical/running components, inferred from text and any visible engine bay photos."
    )

    # --- Specific damage/issue indicators (None = no evidence either way) ---
    accident_indicated: bool | None = Field(
        default=None, description="True if photos or text indicate a prior accident/collision."
    )
    rust_or_corrosion: bool | None = Field(
        default=None, description="True if rust or corrosion is visible or mentioned."
    )
    paint_or_body_issues: bool | None = Field(
        default=None, description="True if there are paint defects, dents, scratches, or panel misalignment."
    )
    paint_or_body_notes: str = Field(
        default="", description="Short free-text note on the specific paint/body issue, empty string if none observed."
    )
    tire_condition: str = Field(
        default="unknown", description="Short description of tire condition/tread, e.g. 'good, even wear' or 'bald front tires'."
    )
    glass_and_lights_condition: str = Field(
        default="unknown", description="Short description of glass/headlight/taillight condition, e.g. 'cracked windshield'."
    )
    warning_lights: list[str] = Field(
        default_factory=list, description="Dashboard warning lights mentioned or visible, e.g. ['check engine', 'ABS']."
    )
    modifications: list[str] = Field(
        default_factory=list, description="Aftermarket modifications mentioned or visible, e.g. ['aftermarket exhaust', 'lift kit']."
    )
    missing_or_damaged_components: list[str] = Field(
        default_factory=list, description="Specific components noted as missing, broken, or damaged."
    )

    water_or_flood_indicators: bool | None = Field(
        default=None, description="True if there are signs of water/flood damage (musty smell mentioned, water lines, corrosion pattern)."
    )
    smoke_or_pet_indicators: bool | None = Field(
        default=None, description="True if there are signs of smoking or pet use (odor mentioned, stains, hair, burn marks)."
    )
    signs_of_neglect: bool | None = Field(
        default=None, description="True if the vehicle appears poorly maintained (heavy grime, neglected interior, overdue service mentioned)."
    )
    signs_of_recent_repair: bool | None = Field(
        default=None, description="True if there is evidence of recent repair work (fresh paint on one panel, new parts, mismatched panels)."
    )

    undisclosed_issues: list[str] = Field(
        default_factory=list,
        description="Issues visible in photos that are NOT mentioned in the seller's description text — the discrepancy is the signal.",
    )
    positive_indicators: list[str] = Field(
        default_factory=list, description="Positive condition signals worth surfacing, e.g. ['new tires', 'garage kept', 'single owner']."
    )

    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in this assessment, 0.0 (guessing) to 1.0 (certain)."
    )
    evidence: str = Field(
        min_length=1,
        description="Reasoning for this assessment, citing specific things seen in the photos or read in the description text. Required for auditability.",
    )

    @classmethod
    def ollama_response_format(cls) -> dict:
        """
        JSON schema to pass as the `format` field of an Ollama /api/chat
        request body, constraining the model's output to this shape.
        """
        return cls.model_json_schema()
