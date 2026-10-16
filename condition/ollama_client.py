"""
condition/ollama_client.py

Thin httpx-based client for Ollama's /api/chat endpoint, used to run the
vision-language condition extraction. Deliberately does NOT depend on the
`ollama` pip package — httpx is already this project's HTTP library
(scrapers/Ingest.py etc. all go through it), so talking to Ollama's plain
REST API with httpx keeps one fewer dependency and one consistent way of
doing HTTP in this codebase.

Ollama's /api/chat request/response shape (relevant subset):

    POST {OLLAMA_HOST}/api/chat
    {
      "model": "qwen2.5vl:7b",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "...", "images": ["<base64>", ...]}
      ],
      "format": {...json schema...},
      "options": {"temperature": 0.0, "num_ctx": 8192},
      "stream": false
    }

    -> {"message": {"role": "assistant", "content": "<json string>"}, ...}

`stream: false` is required — the default is streaming NDJSON, which is not
what we want for a single structured-output parse.
"""
from __future__ import annotations

import json
import logging
import time

import httpx
from pydantic import ValidationError

from config import (
    OLLAMA_HOST,
    QWEN_MODEL,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_RETRIES,
)
from condition.schema import ConditionAssessmentResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert vehicle condition inspector. You will be given a "
    "vehicle listing's description text and, optionally, one or more photos "
    "from the listing. Assess the vehicle's condition strictly from what is "
    "stated in the text or visible in the photos — do not guess about things "
    "you cannot see or that are not mentioned. If something cannot be "
    "determined, say so (use 'unknown' for ratings, null for indicator "
    "booleans, empty list for list fields) rather than fabricating detail. "
    "Respond only with the requested structured JSON."
)

# Transient HTTP status codes worth retrying (server-side/overload errors).
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

# Backoff schedule (seconds) between retry attempts — short, since this runs
# inside a batch backfill processing thousands of listings and Ollama is a
# local/LAN service, not a rate-limited third party API.
_BACKOFF_BASE_SECONDS = 1.5


class OllamaUnavailableError(Exception):
    """Raised when Ollama could not produce a valid structured response after
    OLLAMA_MAX_RETRIES attempts (connection errors, 5xx, or malformed/invalid
    JSON output every time). Callers should catch this specifically and
    degrade gracefully (mark the extraction as failed) rather than crash."""


def _build_user_content(description: str, image_b64_list: list[str]) -> str:
    if description:
        return (
            "Vehicle listing description:\n"
            f"{description}\n\n"
            "Assess this vehicle's condition using the description above and "
            f"the {len(image_b64_list)} attached photo(s), if any."
        )
    return (
        "No description text was provided for this listing. Assess this "
        f"vehicle's condition using only the {len(image_b64_list)} attached "
        "photo(s)."
    )


def _post_chat(payload: dict) -> httpx.Response:
    url = f"{OLLAMA_HOST}/api/chat"
    with httpx.Client(timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as client:
        return client.post(url, json=payload)


def extract_condition(description: str, image_b64_list: list[str]) -> ConditionAssessmentResult:
    """
    Calls Ollama's chat API with the given description text + images and
    parses/validates the structured response into a ConditionAssessmentResult.

    Retries up to OLLAMA_MAX_RETRIES times (with a short backoff) on:
      - connection errors (Ollama not reachable / connection reset)
      - HTTP 5xx responses
      - responses whose body is not valid JSON, or doesn't validate against
        the ConditionAssessmentResult schema

    Raises OllamaUnavailableError if every attempt fails. Never returns a
    partially-valid result — callers get either a fully validated
    ConditionAssessmentResult or an exception.
    """
    description = description or ""
    image_b64_list = image_b64_list or []

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_content(description, image_b64_list),
                "images": image_b64_list,
            },
        ],
        "format": ConditionAssessmentResult.ollama_response_format(),
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        },
        "stream": False,
        "think": False,
    }

    last_error: Exception | None = None
    max_attempts = max(1, OLLAMA_MAX_RETRIES)

    for attempt in range(1, max_attempts + 1):
        try:
            response = _post_chat(payload)
        except httpx.TransportError as exc:
            # Connection refused/reset, DNS failure, timeout, etc.
            last_error = exc
            logger.warning(
                "Ollama request failed (attempt %d/%d): %s: %s",
                attempt, max_attempts, type(exc).__name__, exc,
            )
            _sleep_backoff(attempt)
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES:
            last_error = httpx.HTTPStatusError(
                f"Ollama returned transient status {response.status_code}",
                request=response.request,
                response=response,
            )
            logger.warning(
                "Ollama returned retryable status %d (attempt %d/%d)",
                response.status_code, attempt, max_attempts,
            )
            _sleep_backoff(attempt)
            continue

        if response.status_code != 200:
            # Non-transient client error (bad request, model not found, etc.)
            # — retrying won't help, but we still exhaust attempts for
            # simplicity/consistency and to tolerate flaky proxies; the error
            # is captured either way.
            last_error = httpx.HTTPStatusError(
                f"Ollama returned status {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
            logger.warning(
                "Ollama returned non-OK status %d (attempt %d/%d)",
                response.status_code, attempt, max_attempts,
            )
            _sleep_backoff(attempt)
            continue

        try:
            body = response.json()
            content = body["message"]["content"]
            parsed = json.loads(content)
            result = ConditionAssessmentResult.model_validate(parsed)
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Ollama response was not valid structured output (attempt %d/%d): %s: %s",
                attempt, max_attempts, type(exc).__name__, exc,
            )
            _sleep_backoff(attempt)
            continue

        return result

    logger.error(
        "Ollama extraction failed after %d attempt(s), giving up: %s",
        max_attempts, last_error,
    )
    raise OllamaUnavailableError(
        f"Ollama did not return a valid structured response after {max_attempts} attempt(s): {last_error}"
    ) from last_error


def _sleep_backoff(attempt: int) -> None:
    time.sleep(_BACKOFF_BASE_SECONDS * attempt)
