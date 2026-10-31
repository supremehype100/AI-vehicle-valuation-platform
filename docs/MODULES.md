# Modules and components

The project is built from small parts with clear jobs. Each can be understood,
tested and replaced on its own. This page describes every part: what it does,
what goes in and out, what it depends on and how to use it directly.

- [Dependency map](#dependency-map)
- [Core](#core): `config`, `models`, `Session`
- [Data collection](#data-collection): scrapers
- [Data processing](#data-processing): `Ingest`, `classification`, `Features`
- [AI condition extraction](#ai-condition-extraction): `condition`, `backfill`
- [Modelling](#modelling): `Model`, `training`
- [Operations](#operations): `migrations`, `run.py`, `Scheduler.py`, `scripts`
- [Tests](#tests)

## Dependency map

Arrows point from a module to what it imports. Lower layers never import upper ones.

```
                 run.py ─────────────► Scheduler.py (calls run.py as subprocesses)
                   │
   ┌───────────────┼───────────────────────────────────────────┐
   ▼               ▼                   ▼                       ▼
 scrapers        Ingest            backfill/                training/ ── Model.py
 kijiji.py         │                  │                        │
 Autotrader.py     ├─► classification │                        │
 Facebook.py       │                  ▼                        │
 scrapers/statvin  │              condition/                   │
   │               │         (schema, image_utils,             │
   │               │          ollama_client, extract, merge)   │
   │               │                  │                        │
   └──► Features ◄─┴──────────────────┴────────────────────────┘
          │
          ▼
   models.py ◄── Session.py ◄── config.py        (migrations/ talks to the DB directly)
```

Rules of thumb that keep the parts independent:

- **Scrapers know nothing about the database.** They yield plain dictionaries.
- **Ingest is the only writer of listings.** Everything that stores a listing goes through it.
- **`classification/classify.py` is the only place that decides a listing's type.**
- **`config.py` is the only place that reads environment variables.**
- **Training reads through `Features`/`models`, never from scrapers.**

---

## Core

### `config.py`: settings
- **Does:** reads `.env` and the environment, validates numbers, defines constants (listing types, precedence levels, Ollama and image settings).
- **Side effects:** creates `MODEL_DIR` and `MODEL_REGISTRY_DIR`. No network, no database.
- **Use it:** `from config import DATABASE_URL, QWEN_MODEL, LISTING_TYPE_PRIVATE`
- **Reference:** [CONFIGURATION.md](CONFIGURATION.md)

### `models.py`: data shapes
- **Pydantic:** `RawListing`, `ListingSnapshot`, lightweight validation shapes.
- **SQLAlchemy tables:** `Listing`, `DBListingSnapshot`, `PricePrediction`, `TrainingRun`, `ConditionAssessment`, `ModelArtifact`.
- **Also:** `Listing.make_fingerprint(year, make, model, mileage, city)` builds the duplicate-detection hash (mileage rounded to 5,000 km).
- **Reference:** [DATABASE.md](DATABASE.md)

### `Session.py`: database access
- `create_tables()` creates all ORM tables if missing.
- `get_session()` is a context manager. It commits on success, rolls back on error and always closes.

```python
from Session import get_session
from models import Listing
from sqlalchemy import select

with get_session() as session:
    cheap = session.execute(select(Listing).where(Listing.price < 5000)).scalars().all()
```

---

## Data collection

All four scrapers follow the same contract, which is what makes them
interchangeable:

| Piece | Signature | Meaning |
| --- | --- | --- |
| `normalize(raw)` | `dict -> dict \| None` | Turn one site-specific record into the **standard listing dict**, or `None` to drop it (bad price, missing year/make, parts car...). |
| `scrape_page(page, client)` | `-> list[dict]` | Fetch and normalize one results page. |
| `run(max_pages, known_ids)` | generator of dicts | Walk pages politely, skipping ids you already have, stopping early when a page has nothing new. |

### The standard listing dict
Every scraper returns dictionaries with (at least) these keys, which `Ingest.upsert` understands:

```
source, source_id, url, vin
title, make, model, year, mileage, trim, color
price, description, status ("active"), seller_type, seller_id, location
image_urls (list), raw_json (dict, extras for ML), posted_at
# salvage only: damage_primary, damage_secondary, title_type, auction_house,
#               auction_date, lot_number, run_drive_status, auction_sale_status
```

`source` + `source_id` together are the unique key.

### Kijiji scraper
- **File:** `kijiji.py`. **Method:** plain HTTP (`httpx`), reading the `__NEXT_DATA__` JSON that Kijiji embeds in each results page. Listings sit in an Apollo cache under keys such as `AutosListing:<id>`.
- **Scope:** Calgary, private sellers only (`for-sale-by=ownr`). Change `SEARCH_URL` for another place.
- **Extras:** fetches the full description for truncated ads; `clean_description()` strips boilerplate (also reused by AutoTrader).
- **Politeness:** random 1-2.5 s sleeps.
- **Try it:**
  ```python
  from kijiji import run
  for listing in run(max_pages=1):
      print(listing["year"], listing["make"], listing["model"], listing["price"])
  ```

### AutoTrader scraper
- **File:** `Autotrader.py`. **Method:** HTTP + embedded Next.js JSON, searched recursively for listing objects.
- **Scope:** Calgary, private sellers, 100 per page, damaged listings included. Note that damaged vehicles are labelled private here; filter on description flags if that matters to you.
- **Politeness:** random 2-4 s sleeps.

### Facebook Marketplace scraper
- **File:** `Facebook.py`. **Method:** a real Chromium window driven by Playwright with a **persistent, already-logged-in profile** (`./fb_chrome_profile_fresh`). It collects item links from a feed page, opens each, reads the visible text and passes it through `normalize_visible()`.
- **Why a browser:** Marketplace needs a logged-in session and renders in JavaScript. It runs headed (a window opens), so it is **not a good fit for Docker** and not for headless servers.
- **Setup and risks:** [ADVANCED_SETUP.md](ADVANCED_SETUP.md#facebook-marketplace-setup). Automated access may breach Facebook's terms; use your own account and low volume.
- **Feeds:** `run.FB_TARGET_URLS` lists the search pages to visit; `cmd_scrape_facebook` processes them one at a time.
- **Timing:** governed by the `FB_*` settings.

### stat.vin scraper
- **File:** `scrapers/statvin.py`. **Source:** a public aggregator of Copart/IAAI salvage lots. Produces `listing_type="salvage"` rows with damage, title type, auction house, lot number, drive status and current bid.
- **Method:** parses rendered HTML text (there is no embedded JSON).
- **Honest status:** the parsing and `normalize()` are unit-tested, but the HTTP transport was **never verified against the live site** (it was blocked by Cloudflare during development). If you get `403` errors, only `_fetch_page()` needs a different transport; all parsing is transport-agnostic.
- **Important limitation:** it records `auction_sale_status="pending"` (current bid, not a final price), and Model 4 trains only on rows with status `sold`. Until something populates `sold` rows, `train_salvage_traintest` will report not enough data.

---

## Data processing

### `Ingest.py`: the single writer
| Function | Purpose |
| --- | --- |
| `upsert(listing_dict, session)` | Insert or update one listing. Returns `'new'`, `'updated'` or `'unchanged'`. Normalizes make/model casing, classifies the listing, computes the fingerprint, writes a snapshot when price or status changed. |
| `upsert_batch(listings, session)` | Runs `upsert` for many listings. Each listing gets its own database savepoint, so one bad row is counted as an error and does not roll back the others. Returns `{new, updated, unchanged, errors}` counts. |
| `get_known_ids(source, session)` | Set of `source_id`s already stored; passed to scrapers so they skip known ads. |
| `sold_sweep(source, seen_ids, session)` | After a **full** scrape, marks active listings that were not seen as `removed`, records `days_on_market` and a final snapshot. Does nothing if `seen_ids` is empty (a failed scrape must not look like "everything sold"). |

### `classification/classify.py`: private, dealer or salvage?
`classify_listing(listing_dict) -> "private" | "dealer" | "salvage"`, in this order:

1. source is a salvage source (`statvin`) → `salvage`
2. has auction fields (`damage_primary` or `lot_number`) → `salvage`
3. `seller_type == "dealer"` → `dealer`
4. seller name matches dealer words (auto, motors, dealership, leasing...) or the description uses sales language (financing available, warranty included, certified pre-owned...) → `dealer`
5. otherwise → `private`

`dealer_match_reason()` returns the human-readable reason. The patterns are heuristics and will misclassify some ads. For example, a private seller who writes "test drive welcome" matches the description pattern.

### `Features.py`: from listings to model inputs
| Function | Purpose |
| --- | --- |
| `is_valid_price(price)` | $1,000 to $200,000 and not a placeholder such as 12345. |
| `extract_transmission(description, raw)` | `manual` or `automatic`, defaulting to automatic. |
| `extract_description_flags(description)` | 0/1 flags from regexes: positive (`accident_free`, `one_owner`, `new_tires`, `recent_service`, `no_rust`...) and negative (`rebuilt`, `salvage`, `needs_work`, `rust`, `inop`, `parts_car`...). |
| `listings_to_dataframe(listings)` | Listings → pandas table; skips invalid prices, missing years, parts cars and inoperable cars. |
| `load_training_data(session, listing_type=...)` | Query + dataframe. A `listing_type` is required in practice so datasets never mix. |
| `prepare_features(df, label_encoders, fit)` | Label-encodes the categorical columns, with an `unknown` bucket for unseen values. Returns `(X, y, encoders)`. |

---

## AI condition extraction

An **optional** subsystem that turns free text and photos into structured condition data. The result is stored and queryable, but it is not yet a model input. See [MODELS.md](MODELS.md#what-the-models-use-today).

### `condition/` pieces
| File | Role |
| --- | --- |
| `schema.py` | `ConditionAssessmentResult`, the strict JSON shape the AI must return: overall/exterior/interior/mechanical ratings (`excellent`, `good`, `fair`, `poor`, `unknown`), yes/no/unknown indicators (accident, rust, flood, smoke/pet, neglect, recent repair), evidence list, notes, confidence. Passed to Ollama as a JSON schema so the output is machine-readable. |
| `image_utils.py` | Downloads photos, validates, resizes (≤ `IMAGE_MAX_DIMENSION_PX`), drops near-duplicate photos using a perceptual hash, keeps up to `MAX_IMAGES_PER_LISTING`. |
| `ollama_client.py` | Calls Ollama's `/api/chat` over `httpx` (no extra SDK), with retries and backoff; raises `OllamaUnavailableError` if the server is down. |
| `extract.py` | `extract_condition_for_listing(description, image_urls)` runs the full flow for one listing and always returns a dict (failures become `extraction_status="failed"` with a message instead of exceptions). |
| `merge.py` | `merge_condition_fields(...)` decides which values may overwrite the listing's current condition using the **precedence rule**. |

### Precedence rule
Each condition field remembers where its value came from. A new value overwrites
the stored one only if its tier is **greater than or equal to** the stored tier:

```
manually validated (40)  >  trusted structured data (30)  >  AI extracted (20)  >  weak inference (10)
```

So re-running the AI can refresh its own earlier output but can never overwrite
something you verified by hand. Tests: `tests/test_condition_merge.py`.

### Audit trail
Every attempt, successful or not, inserts a row into `condition_assessments`
(model version, schema version, which photos, a hash of the text, the raw
output and confidence). The `listings.condition_*` columns hold the current
merged view.

### `backfill/backfill_conditions.py`: run it over the database
```bash
python -m backfill.backfill_conditions [--dataset private|dealer|salvage|all]
                                       [--batch-size 50] [--limit N] [--resume]
```
Resumable with no checkpoint file: a listing drops out of the selection query
once it has a confidence value and an assessment row for the **current** model
and schema version. Kill it and re-run the same command to continue. It prints
aggregate counts only, never listing contents.

---

## Modelling

### `Model.py`: the baseline model and valuation
- `train(session, cv_csv_path=None)` trains Model 1 and writes `xgb_price_model.pkl`, `label_encoders.pkl`, `model_meta.pkl` into `MODEL_DIR`, plus a `training_runs` row.
- `valuate(session, make, model_name, year, mileage, ...)` loads those files and returns the predicted price, a ±15% range (tightened by comparable listings), the comparables and the model's MAE.
- `score_listing(session, listing)` values an existing listing and stores a `price_predictions` row.

### `training/`: the four-model system
| File | Role |
| --- | --- |
| `private_csv.py` | Model 1; wraps `Model.train` and registers the result. |
| `private_traintest.py`, `dealer_traintest.py` | Models 2 and 3; thin wrappers over `_traintest_common.run_traintest` with a different `listing_type`. |
| `salvage_traintest.py` | Model 4; its own salvage feature set (damage, title type, drive status, auction house). |
| `_traintest_common.py` | Shared split → encode → train → score → save → register logic. |
| `splits.py` | De-duplicate by fingerprint, time-aware split, VIN-disjoint enforcement and a final leakage assertion. |
| `leakage.py` | Documents which features are knowable at prediction time and zeroes out `days_on_market`. |
| `metrics.py` | `compute_metrics`: MAE, RMSE, R², median error, MAPE, bias, share within ±$500/$1k/$2.5k/$5k, error percentiles. |
| `registry.py` | `register_model`, `get_latest`, `list_versions` over the `model_artifacts` table. |
| `fingerprint.py` | Hashes identifying the exact dataset a model was trained on. |
| `evaluate.py` | Comparison report and plots (actual vs predicted, residuals, error by bucket, error over time). |
| `private_csv.py` / `scripts/evaluate_private_csv.py` | Evaluate against your own CSV, printing aggregate numbers only. |

Deep dive: [MODELS.md](MODELS.md).

---

## Operations

### `migrations/`
Numbered plain-SQL files plus `migrate.py`, a small runner that applies pending
files in order and records them in `schema_migrations`. See
[DATABASE.md](DATABASE.md#migrations). `python run.py init_db` creates the
tables and then runs it.

### `run.py`: the command line
Run `python run.py <command>`. Commands that take a page count accept it as the second argument.

| Group | Commands |
| --- | --- |
| Setup | `init_db`, `status` |
| Collect | `scrape [pages]`, `scrape_fast [pages]`, `bulk [pages]`, `scrape_facebook`, `scrape_statvin [pages]`, `sold_sweep`, `backfill_kijiji` |
| Understand | `backfill_conditions` |
| Model | `train`, `train_private_csv`, `train_private_traintest`, `train_dealer_traintest`, `train_salvage_traintest`, `train_all`, `compare_models`, `compare_private_fair` |
| Use | `value`, `pipeline [pages]` |

`pipeline` = scrape then train. `train_all` skips (and reports) any model without enough data.
Defaults when you give no page count: `scrape` 5, `scrape_fast` 1, `bulk` 100.

### `Scheduler.py`: run on a timetable
A long-running process with worker threads that call `run.py` as subprocesses so a crash in one job cannot take the scheduler down. Timetable in [ADVANCED_SETUP.md](ADVANCED_SETUP.md#running-on-a-schedule).

### `scripts/`
`evaluate_private_csv.py` scores the saved baseline model against a private CSV and prints only aggregate statistics.

---

## Tests

`tests/` holds 16 test files and 127 tests, none of which need a database, a
network connection or a GPU (database access is mocked).

```bash
python -m pytest -q
```

| Area | Files |
| --- | --- |
| Classification and ingest | `test_classification.py`, `test_ingest_classification.py`, `test_fingerprint.py` |
| Features | `test_features_listing_type_filter.py` |
| Scrapers | `test_statvin_normalize.py`, `test_regressions.py` (Facebook parsing) |
| Condition AI | `test_condition_schema.py`, `test_condition_merge.py`, `test_backfill_logic.py`, `test_backfill_report_format.py` |
| Modelling | `test_splits.py`, `test_metrics.py`, `test_registry.py`, `test_fair_comparison.py` |
| Config and infra | `test_delay_config.py`, `test_docker_compose.py` |

To add a component or swap one out, see [EXTENDING.md](EXTENDING.md).
