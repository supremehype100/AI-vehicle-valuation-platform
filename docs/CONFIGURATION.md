# Configuration reference

Everything tunable is read from **environment variables** by `config.py`. You
can set them three ways, in increasing priority:

1. Built-in defaults (what `config.py` falls back to).
2. A `.env` file next to `config.py` (copy `.env.example` to `.env`).
3. Real environment variables (`export NAME=value`, `$env:NAME="value"`, or the
   `environment:` block in `docker-compose.yml`). **These always win** over
   the `.env` file.

The `.env` parser is deliberately tiny: one `NAME=value` per line, `#` starts a
comment line, and surrounding quotes are stripped. No variable expansion
(`${OTHER}`) and no inline comments after a value.

Invalid numbers fail fast at import time with a message such as
`Environment variable OLLAMA_NUM_CTX='abc' is not a valid int`.

---

## Database

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/vehicle_intel` | SQLAlchemy URL. A leading `postgresql://` is automatically rewritten to `postgresql+psycopg2://` because newer SQLAlchemy versions would otherwise pick a driver this project does not install. |
| `POSTGRES_PASSWORD` | `postgres` | Used by `docker-compose.yml` to set the container's password **and** build the app's `DATABASE_URL`. Change it for anything beyond a local sandbox. |
| `POSTGRES_USER` | `postgres` | Compose only. |
| `POSTGRES_DB` | `vehicle_intel` | Compose only. |

> When you use Docker Compose, the `app` service ignores `DATABASE_URL` from
> `.env` and builds its own pointing at the `db` container. Set
> `POSTGRES_PASSWORD` instead.

## Model files

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_DIR` | `models` | Where trained model files (`.pkl`) are written. Relative paths are relative to the **current working directory**. Created automatically. |
| `MODEL_REGISTRY_DIR` | `<MODEL_DIR>/registry` | Currently only created, not read by any module (the registry itself lives in the database). Safe to ignore. |

## Facebook Marketplace scraper timing

| Variable | Default | Description |
| --- | --- | --- |
| `FB_LISTING_DELAY_MIN_SECONDS` | `4.5` | Minimum random pause between visiting listings. |
| `FB_LISTING_DELAY_MAX_SECONDS` | `7.5` | Maximum pause. Must be ≥ the minimum. |
| `FB_PAGE_LOAD_WAIT_MIN_MS` | `5000` | Minimum wait for a listing page to render before reading it. |
| `FB_PAGE_LOAD_WAIT_MAX_MS` | `7000` | Maximum wait. |

Lowering these makes scraping faster and more likely to be rate-limited or
flagged. Raise them if Facebook starts showing checkpoints.

## AI condition extraction (Ollama)

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the Ollama server listens. In Compose it is `http://ollama:11434`. |
| `QWEN_MODEL` | `qwen3-vl:4b-thinking` | Ollama model tag. Must be a **vision-language** model. Changing it changes the stored provenance string (`ollama:<tag>`), so a backfill re-processes listings under the new model. |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `120` | Per-request timeout. Raise it for CPU-only inference. |
| `OLLAMA_NUM_CTX` | `8192` | Context window passed to Ollama. More context uses more VRAM. |
| `OLLAMA_TEMPERATURE` | `0` | Keep at `0` so re-runs are reproducible. |
| `OLLAMA_MAX_RETRIES` | `3` | Retries with backoff when a request fails or returns invalid JSON. |
| `OLLAMA_USE_GPU` | `true` | Parsed by `config.py` and passed through by Compose, but no module currently branches on it. GPU passthrough is controlled by the Compose files. See [GPU_SETUP.md](GPU_SETUP.md). |

## Image handling

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_IMAGES_PER_LISTING` | `6` | Photos sent to the model per listing, after de-duplication. |
| `IMAGE_DOWNLOAD_TIMEOUT_SECONDS` | `15` | Timeout for each photo download. |
| `IMAGE_MAX_DIMENSION_PX` | `1024` | Photos are shrunk so the longest side is at most this. Smaller means faster and less VRAM. |
| `IMAGE_CACHE_DIR` | `./image_cache` | Local cache of downloaded photos (git-ignored). |

## Scheduler

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_FACEBOOK_SCHEDULE` | unset (off) | Set to `1`/`true`/`yes`/`on` to let `Scheduler.py` also run the Facebook scrape every 3 hours. Off by default because it needs a logged-in browser profile and a display. |

---

## Constants that live in code

These are not environment variables. Edit the source to change them.

| What | Where | Value |
| --- | --- | --- |
| Valid price window | `Features.is_valid_price` | $1,000 to $200,000, excluding placeholders `1234`, `12345`, `123456` |
| Minimum rows to train | `Model.MIN_TRAINING_SAMPLES`, `training/_traintest_common.py`, `training/salvage_traintest.py` | 50 |
| Test fraction | `run(..., test_fraction=0.15)` in each `training/*_traintest.py` | 0.15 |
| XGBoost hyper-parameters | `Model.train()` and `training/_traintest_common.XGB_PARAMS` | 500 trees, learning rate 0.05, depth 6, early stopping 30 |
| Valuation range | `Model.valuate()` | ±15% of the prediction, tightened to the 15th-85th percentile of comparables when at least 3 exist |
| Comparable matching | `Model._find_comparables()` | same make/model, ±2 model years, ±30,000 km, active only |
| Kijiji search URL | `kijiji.SEARCH_URL` | Calgary, private sellers only |
| AutoTrader search URL | `Autotrader.SEARCH_URL` | Calgary, private sellers, damaged listings included, 100 per page |
| Facebook feeds | `run.FB_TARGET_URLS` | 12 Calgary Marketplace feed URLs |
| Facebook profile folder | `Facebook.CHROME_USER_DATA` | `./fb_chrome_profile_fresh` |
| Scraper politeness delays | `kijiji.py`, `Autotrader.py`, `scrapers/statvin.py` | 1-2.5 s, 2-4 s and 2-4 s random sleeps between requests |
| Scheduler timetable | `Scheduler.py` | see [ADVANCED_SETUP.md](ADVANCED_SETUP.md#running-on-a-schedule) |
| Condition precedence | `config.PRECEDENCE_*` | manual 40 > structured 30 > AI 20 > weak inference 10 |
| Listing types | `config.LISTING_TYPE_*` | `private`, `dealer`, `salvage` |

## Files and folders the project creates

| Path | Created by | Git-ignored | Purpose |
| --- | --- | --- | --- |
| `.env` | you | yes | Your settings and secrets |
| `models/` | training | yes (`*.pkl`, `*.json`) | Trained model files |
| `image_cache/` | condition extraction | yes | Cached photos |
| `reports/` | `compare_models` | yes | Markdown comparison report and plots |
| `fb_chrome_profile_fresh/` | Facebook scraper | yes | Logged-in browser profile. **Treat like a password.** |
| `fb_session/` | optional / legacy | yes | Facebook session data. Same warning. |
| `vehicles_cleaned.csv` | you (optional) | yes (`*.csv`) | Your own validation data for Model 1 |

## Security checklist

- Keep `.env`, `fb_chrome_profile_fresh/`, `fb_session/` and any `*.csv` out of
  Git. The `.gitignore` and `.dockerignore` already exclude them; do not force-add.
- Use a non-default `POSTGRES_PASSWORD` the moment the database is reachable by
  anyone other than you. Compose publishes Postgres on host port `5432`; bind it
  to localhost or remove the `ports:` entry for anything non-local
  (see [ADVANCED_SETUP.md](ADVANCED_SETUP.md#securing-a-non-local-deployment)).
- If you ever commit a secret by mistake, rotate it. Deleting it in a later
  commit does not remove it from history.
