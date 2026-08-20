"""
condition/extract.py

Top-level orchestrator that turns a listing's raw fields (description text +
image URLs) into a dict shaped exactly like the `ConditionAssessment` table
columns (models.py), ready for a caller (backfill/backfill_conditions.py) to
INSERT as-is.

This function is designed to be called across thousands of listings in a
batch job and must NEVER raise — every failure mode (no usable input,
Ollama unreachable, bad model output) is caught and turned into a
`extraction_status="failed"` row with a human-readable `error_message`, so a
backfill loop can log-and-continue instead of crashing partway through a run.
"""
from __future__ import annotations

import hashlib
import logging

from config import CONDITION_MODEL_VERSION, CONDITION_SCHEMA_VERSION
from condition import image_utils
from condition.ollama_client import extract_condition, OllamaUnavailableError

logger = logging.getLogger(__name__)


def _description_hash(description: str | None) -> str:
    return hashlib.sha256((description or "").encode("utf-8")).hexdigest()


def extract_condition_for_listing(description: str | None, image_urls: list[str]) -> dict:
    """
    Runs the full condition-extraction pipeline for one listing's text +
    photos and returns a dict with exactly the fields needed to construct a
    `ConditionAssessment` row:

        model_version, schema_version, images_used, description_hash,
        extraction_status ("success"|"failed"), error_message,
        raw_extraction (dict), confidence_overall

    Never raises. On any failure, returns extraction_status="failed" with
    error_message populated and raw_extraction={}/confidence_overall=None.
    """
    description = description or None
    image_urls = image_urls or []
    description_hash = _description_hash(description)

    base_result = {
        "model_version": CONDITION_MODEL_VERSION,
        "schema_version": CONDITION_SCHEMA_VERSION,
        "images_used": [],
        "description_hash": description_hash,
        "extraction_status": "failed",
        "error_message": None,
        "raw_extraction": {},
        "confidence_overall": None,
    }

    if not description and not image_urls:
        base_result["error_message"] = "No description text and no image URLs available for this listing."
        return base_result

    try:
        base64_images, provenance = image_utils.select_images_for_listing(image_urls)
    except Exception as exc:  # noqa: BLE001 - image pipeline must never crash the batch
        logger.warning("Image selection failed: %s: %s", type(exc).__name__, exc)
        base64_images, provenance = [], []
    base_result["images_used"] = provenance

    if not description and not base64_images:
        base_result["error_message"] = (
            "No description text and no usable images (all image downloads/decodes failed)."
        )
        return base_result

    try:
        result = extract_condition(description or "", base64_images)
    except OllamaUnavailableError as exc:
        base_result["error_message"] = f"Ollama unavailable: {exc}"
        return base_result
    except Exception as exc:  # noqa: BLE001 - defensive: any unexpected error must not crash the batch
        logger.exception("Unexpected error during condition extraction")
        base_result["error_message"] = f"Unexpected extraction error: {type(exc).__name__}: {exc}"
        return base_result

    base_result["extraction_status"] = "success"
    base_result["raw_extraction"] = result.model_dump()
    base_result["confidence_overall"] = result.confidence
    return base_result
