"""
condition/image_utils.py

Image handling for the condition-extraction pipeline: downloading listing
photos, cheaply deduping near-identical shots (marketplaces frequently repeat
the same photo, or near-identical burst shots, across a listing's gallery),
resizing/re-encoding for the vision model, and producing a small provenance
record of what was actually sent.

Dedup approach: an 8x8 grayscale "average hash" (aHash) — resize to 8x8,
grayscale, threshold each pixel against the mean pixel value to get a 64-bit
fingerprint, then compare candidate hashes by Hamming distance. This is a
well-known cheap perceptual hash: no ML model, pure Pillow, good enough to
catch "literally the same photo" and "near-identical burst shot" without
false-positiving on genuinely different angles of the same car.

Everything here degrades gracefully — a broken/missing image URL should never
crash the pipeline, it should just be skipped and recorded as such.
"""
from __future__ import annotations

import base64
import io
import logging

import httpx
from PIL import Image

from config import (
    MAX_IMAGES_PER_LISTING,
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
    IMAGE_MAX_DIMENSION_PX,
)

logger = logging.getLogger(__name__)

# Hamming distance (out of 64 bits) at or below which two images are treated
# as near-duplicates. Small enough to avoid merging genuinely different
# angles/shots, large enough to catch re-compressed/re-cropped copies of the
# same photo (a common pattern when a scraper picks up both a thumbnail and
# full-size version of the same image).
HASH_DEDUPE_THRESHOLD = 5

JPEG_QUALITY = 85


def download_image(url: str) -> bytes | None:
    """
    Downloads raw image bytes from `url`. Returns None (and logs a warning,
    never raises) on any failure: network error, timeout, non-2xx status, or
    a non-image content-type. Callers must treat None as "skip this image".
    """
    if not url:
        return None
    try:
        with httpx.Client(timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.lower().startswith("image/"):
            logger.warning("Skipping non-image content-type %r for %s", content_type, url)
            return None
        return response.content
    except httpx.HTTPStatusError as exc:
        logger.warning("Image download got status %s for %s", exc.response.status_code, url)
        return None
    except httpx.HTTPError as exc:
        logger.warning("Image download failed (%s) for %s", type(exc).__name__, url)
        return None
    except Exception as exc:  # noqa: BLE001 - never let a bad image kill the pipeline
        logger.warning("Unexpected error downloading %s: %s: %s", url, type(exc).__name__, exc)
        return None


def prepare_image_for_model(raw_bytes: bytes) -> str | None:
    """
    Decodes `raw_bytes`, resizes so its longest dimension is at most
    IMAGE_MAX_DIMENSION_PX (preserving aspect ratio), re-encodes as JPEG, and
    returns a base64-encoded string suitable for Ollama's chat `images` field.
    Returns None on any decode/encode failure.
    """
    if not raw_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        image = image.convert("RGB")

        width, height = image.size
        longest = max(width, height)
        if longest > IMAGE_MAX_DIMENSION_PX:
            scale = IMAGE_MAX_DIMENSION_PX / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - malformed/corrupt image data
        logger.warning("Failed to prepare image for model: %s: %s", type(exc).__name__, exc)
        return None


def _average_hash(raw_bytes: bytes) -> int | None:
    """8x8 grayscale average-hash -> 64-bit int, or None on decode failure."""
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(image.getdata())
        if not pixels:
            return None
        avg = sum(pixels) / len(pixels)
        bits = 0
        for pixel in pixels:
            bits = (bits << 1) | (1 if pixel >= avg else 0)
        return bits
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to hash image: %s: %s", type(exc).__name__, exc)
        return None


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _gather_candidates(image_urls: list[str], max_images: int) -> list[dict]:
    """
    Downloads and hashes candidate images in order, selecting up to
    `max_images` non-near-duplicate images. Stops downloading further
    candidates as soon as `max_images` have been selected.

    Returns a list of internal-use dicts (one per URL attempted):
        {"url": str, "raw_bytes": bytes|None, "hash": int|None,
         "used": bool, "reason_skipped": str|None}

    `raw_bytes` is transient/internal only — callers that persist provenance
    must strip it (see select_images_for_listing).
    """
    kept_hashes: list[int] = []
    candidates: list[dict] = []
    used_count = 0

    for url in image_urls or []:
        if used_count >= max_images:
            break

        raw_bytes = download_image(url)
        if raw_bytes is None:
            candidates.append(
                {"url": url, "raw_bytes": None, "hash": None, "used": False, "reason_skipped": "download_failed"}
            )
            continue

        img_hash = _average_hash(raw_bytes)
        if img_hash is None:
            candidates.append(
                {"url": url, "raw_bytes": None, "hash": None, "used": False, "reason_skipped": "decode_failed"}
            )
            continue

        is_duplicate = any(_hamming_distance(img_hash, kept) <= HASH_DEDUPE_THRESHOLD for kept in kept_hashes)
        if is_duplicate:
            candidates.append(
                {"url": url, "raw_bytes": None, "hash": img_hash, "used": False, "reason_skipped": "near_duplicate"}
            )
            continue

        kept_hashes.append(img_hash)
        used_count += 1
        candidates.append(
            {"url": url, "raw_bytes": raw_bytes, "hash": img_hash, "used": True, "reason_skipped": None}
        )

    return candidates


def select_and_dedupe_images(image_urls: list[str], max_images: int = MAX_IMAGES_PER_LISTING) -> list[str]:
    """
    Downloads each candidate URL, computes an average-hash, drops near
    duplicates (Hamming distance <= HASH_DEDUPE_THRESHOLD of one already
    kept), and returns the URLs of the selected images (order preserved),
    stopping once `max_images` have been selected.
    """
    candidates = _gather_candidates(image_urls, max_images)
    return [c["url"] for c in candidates if c["used"]]


def select_images_for_listing(image_urls: list[str]) -> tuple[list[str], list[dict]]:
    """
    Top-level orchestration: download, dedupe, resize/encode, and produce a
    small provenance record — used by condition/extract.py.

    Returns (base64_list_for_model, provenance_list) where each provenance
    entry is:
        {"url": str, "hash": int|None, "used": bool, "reason_skipped": str|None}

    Provenance entries NEVER contain raw bytes or base64 image data (this is
    persisted long-term in ConditionAssessment.images_used / JSONB, so it
    must stay small).
    """
    candidates = _gather_candidates(image_urls, MAX_IMAGES_PER_LISTING)

    base64_list: list[str] = []
    provenance: list[dict] = []

    for candidate in candidates:
        entry = {
            "url": candidate["url"],
            "hash": candidate["hash"],
            "used": candidate["used"],
            "reason_skipped": candidate["reason_skipped"],
        }
        if candidate["used"]:
            b64 = prepare_image_for_model(candidate["raw_bytes"])
            if b64:
                base64_list.append(b64)
            else:
                entry["used"] = False
                entry["reason_skipped"] = "encode_failed"
        provenance.append(entry)

    return base64_list, provenance
