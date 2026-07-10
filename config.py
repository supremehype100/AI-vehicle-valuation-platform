"""
config.py

Single source of truth for environment-driven configuration. Every module that
previously hardcoded a value that should be tunable (DB URL, model dir, scraper
delays, AI model choice) should import it from here instead of redefining it.

Nothing in this file talks to the network or the database — it only reads
os.environ with sane defaults, so importing it is always safe/cheap.
"""
from __future__ import annotations
import os
from pathlib import Path
# Automatically read the .env file if it exists in the root folder
env_path = Path(__file__).parent.parent / ".env"  # Adjust path depth if needed
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw!r} is not a valid float")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name}={raw!r} is not a valid int")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vehicle_intel",
)

# ---------------------------------------------------------------------------
# Model artifact storage (XGBoost pickles, label encoders, registry)
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True, parents=True)
MODEL_REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", str(MODEL_DIR / "registry")))
MODEL_REGISTRY_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Listing classification — the explicit source/listing type discriminator.
# Do not compare against booleans elsewhere in the codebase; import these.
# ---------------------------------------------------------------------------
LISTING_TYPE_PRIVATE = "private"
LISTING_TYPE_DEALER = "dealer"
LISTING_TYPE_SALVAGE = "salvage"
VALID_LISTING_TYPES = {LISTING_TYPE_PRIVATE, LISTING_TYPE_DEALER, LISTING_TYPE_SALVAGE}

# ---------------------------------------------------------------------------
# Facebook Marketplace scraper delays
#
# There are two distinct waits in Facebook.py:
#   1. Per-listing PAGE LOAD wait inside extract_visible_details() — gives the
#      page time to render before we scrape the DOM. This is a correctness
#      safeguard, not a "pause between checking listings", so it is left alone.
#   2. The pause BETWEEN checking listings (asyncio.sleep between iterations of
#      the listing loop) — this is the one the task asked to reduce to 75% of
#      its original value. Original range was uniform(6.0, 10.0) seconds;
#      75% of that is uniform(4.5, 7.5).
# ---------------------------------------------------------------------------
FB_LISTING_DELAY_MIN_SECONDS = _float_env("FB_LISTING_DELAY_MIN_SECONDS", 4.5)
FB_LISTING_DELAY_MAX_SECONDS = _float_env("FB_LISTING_DELAY_MAX_SECONDS", 7.5)
if FB_LISTING_DELAY_MIN_SECONDS < 0 or FB_LISTING_DELAY_MAX_SECONDS < FB_LISTING_DELAY_MIN_SECONDS:
    raise ValueError(
        "Invalid FB_LISTING_DELAY_MIN_SECONDS/FB_LISTING_DELAY_MAX_SECONDS — "
        f"got min={FB_LISTING_DELAY_MIN_SECONDS}, max={FB_LISTING_DELAY_MAX_SECONDS}"
    )

# Page-load wait (ms) — intentionally NOT reduced, see note above.
FB_PAGE_LOAD_WAIT_MIN_MS = _int_env("FB_PAGE_LOAD_WAIT_MIN_MS", 5000)
FB_PAGE_LOAD_WAIT_MAX_MS = _int_env("FB_PAGE_LOAD_WAIT_MAX_MS", 7000)

# ---------------------------------------------------------------------------
# Ollama / Qwen condition-extraction pipeline
#
# Target hardware is a single GTX 1080 Ti (11GB VRAM). A full-precision or
# lightly-quantized 7B-class vision-language model fits comfortably; the 32B/72B
# Qwen2.5-VL variants do not (they need 20-40GB+ even quantized) and are not
# supported targets for this deployment. "qwen2.5vl:7b" is Ollama's default
# quantization (Q4_K_M, ~6GB on disk / ~7-8GB resident VRAM incl. KV cache),
# which leaves headroom on an 11GB card and still fits an 8GB card in a pinch
# with smaller image batches. Override via QWEN_MODEL if you have more VRAM
# (e.g. "qwen2.5vl:32b") or less (e.g. "qwen2.5vl:3b").
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-vl:4b-thinking")
OLLAMA_REQUEST_TIMEOUT_SECONDS = _float_env("OLLAMA_REQUEST_TIMEOUT_SECONDS", 120.0)
OLLAMA_NUM_CTX = _int_env("OLLAMA_NUM_CTX", 8192)
# temperature=0 for reproducibility — extraction is a structured-output task,
# not a creative one, and the backfill must be re-runnable deterministically.
OLLAMA_TEMPERATURE = _float_env("OLLAMA_TEMPERATURE", 0.0)
OLLAMA_MAX_RETRIES = _int_env("OLLAMA_MAX_RETRIES", 3)

# Combined into stored provenance so we know exactly which config produced a
# given assessment even after QWEN_MODEL is later changed.
CONDITION_MODEL_VERSION = f"ollama:{QWEN_MODEL}"
CONDITION_SCHEMA_VERSION = "condition_v1"

# ---------------------------------------------------------------------------
# Image handling for condition extraction
# ---------------------------------------------------------------------------
MAX_IMAGES_PER_LISTING = _int_env("MAX_IMAGES_PER_LISTING", 6)
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = _float_env("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 15.0)
IMAGE_MAX_DIMENSION_PX = _int_env("IMAGE_MAX_DIMENSION_PX", 1024)
IMAGE_CACHE_DIR = Path(os.environ.get("IMAGE_CACHE_DIR", "./image_cache"))

# ---------------------------------------------------------------------------
# Backfill precedence — higher number wins when merging a field.
# "Do not overwrite trusted historical information with low-confidence AI
# guesses" — this ordering is what enforces that rule everywhere.
# ---------------------------------------------------------------------------
PRECEDENCE_MANUALLY_VALIDATED = 40
PRECEDENCE_TRUSTED_STRUCTURED = 30
PRECEDENCE_AI_EXTRACTED = 20
PRECEDENCE_INFERRED_WEAK = 10

# ---------------------------------------------------------------------------
# Docker / GPU
# ---------------------------------------------------------------------------
OLLAMA_USE_GPU = _bool_env("OLLAMA_USE_GPU", True)
