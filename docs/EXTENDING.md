# Extending and developing

How to change the project safely: add a data source, a feature, a model or a
command, and the conventions to follow. Read [MODULES.md](MODULES.md) first so
the names below make sense.

- [Development workflow](#development-workflow)
- [Add a new data source](#add-a-new-data-source)
- [Add a feature](#add-a-feature)
- [Add a CLI command](#add-a-cli-command)
- [Add a new model or listing type](#add-a-new-model-or-listing-type)
- [Add a database migration](#add-a-database-migration)
- [Change the AI backend or schema](#change-the-ai-backend-or-schema)
- [Conventions](#conventions)

---

## Development workflow

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                       # includes pytest and PyYAML
python -m pytest -q                                   # 127 tests, no DB/network/GPU needed
python -m pyflakes .                                  # optional: unused imports and names
```

The test suite mocks the database, so you can refactor with confidence without
Postgres. Anything touching real SQL, Playwright or Ollama needs a manual run
(see the checklist in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#verifying-a-change-end-to-end)).

---

## Add a new data source

Say you want to add `carsales.example`.

**1. Create the scraper** as `scrapers/carsales.py` (or a top-level file like the
older scrapers). Copy the shape of `kijiji.py`:

```python
from __future__ import annotations
import logging, random, time
from typing import Generator
import httpx
from Features import is_valid_price

logger = logging.getLogger(__name__)
SOURCE = "carsales"
HEADERS = {"User-Agent": "..."}

def normalize(raw: dict) -> dict | None:
    """One site-specific record -> the standard listing dict, or None to drop it."""
    price = ...  # parse
    if not is_valid_price(price):
        return None
    return {
        "source": SOURCE,
        "source_id": str(raw["id"]),      # must be stable and unique per ad
        "url": raw["url"],
        "vin": None,
        "title": raw.get("title"),
        "make": ..., "model": ..., "year": ..., "mileage": ...,
        "price": price,
        "description": raw.get("description"),
        "status": "active",
        "seller_type": None,              # or "dealer" if the site tells you
        "location": raw.get("city"),
        "image_urls": raw.get("images", []),
        "raw_json": {"transmission": ..., "body_type": ..., "fuel_type": ..., "drivetrain": ...},
        "posted_at": raw.get("posted"),   # ISO string is fine
    }

def scrape_page(page: int, client: httpx.Client) -> list[dict]:
    ...  # fetch, parse, return [normalize(r) for r in records if normalize(r)]

def run(max_pages: int = 5, known_ids: set[str] | None = None) -> Generator[dict, None, None]:
    known_ids = known_ids or set()
    with httpx.Client(timeout=20) as client:
        for page in range(1, max_pages + 1):
            listings = scrape_page(page, client)
            if not listings:
                break
            new = 0
            for listing in listings:
                if listing["source_id"] not in known_ids:
                    new += 1
                yield listing
            if new == 0 and known_ids:
                break                       # nothing new: stop early
            time.sleep(random.uniform(2.0, 4.0))   # be polite
```

Rules: no database access inside a scraper, and never hard-code credentials.
Check the site's terms of service and `robots.txt` first.

**2. Wire it into the CLI.** In `run.py`, add a `cmd_scrape_carsales` modelled on
`cmd_scrape_statvin` (`create_tables()`, read `get_known_ids("carsales", session)`,
collect listings, `upsert_batch(...)`) and register it in `COMMANDS`.

**3. Decide the listing type.** Private/dealer needs nothing: the classifier
handles it. If the source is salvage auctions, add its name to `SALVAGE_SOURCES`
in `classification/classify.py`.

**4. Include it in sold detection** (optional): add it to `cmd_sold_sweep` in the
same way as Kijiji and AutoTrader. Only do this if `run(max_pages=100)` really
walks *all* current ads, otherwise listings will be wrongly marked removed.

**5. Schedule it** (optional): add a subprocess call in `Scheduler.py`.

**6. Test `normalize`** with a saved sample record. Follow `tests/test_statvin_normalize.py`.

---

## Add a feature

### A keyword flag from the description
Add a regex to `POSITIVE_SIGNALS` or `NEGATIVE_SIGNALS` in `Features.py`:

```python
POSITIVE_SIGNALS["highway_miles"] = r"highway (km|miles)|mostly highway"
```

It flows into every retail model automatically (the flag dictionary is merged
into each row). Models trained earlier will not know the new column, so
**retrain** (`python run.py train_all`). Also add the new column to
`training/leakage.py`'s `FEATURE_AVAILABILITY` table (documentation only, nothing
enforces it) saying whether it is known at prediction time. A flag derived purely
from the description is.

### A column from the listing
Add it to the row dictionary in `Features.listings_to_dataframe`, and if it is
text, add its name to `CAT_COLS` in `prepare_features`. In `Model.valuate`,
add the same key to the single-row `row` dictionary (training and serving must
produce identical columns or predictions will be wrong).

### AI condition as a model feature
The condition columns already exist on `listings`. To use them:

1. Map ratings to numbers in `Features.listings_to_dataframe`, e.g. `{"excellent": 3, "good": 2, "fair": 1, "poor": 0}` with a missing value for unknown/`NULL`.
2. Add the boolean indicators as 0/1/missing.
3. Mirror every new key in `Model.valuate` (where condition would come from user input or be left missing).
4. Mark them in `training/leakage.py`: they are available at prediction time only if a user can supply them.
5. Retrain and compare against the previous version with `compare_models`.

XGBoost handles missing values natively, so `NaN` for "not assessed" is fine.

---

## Add a CLI command

1. Write `def cmd_something(...)` in `run.py`. Import heavy modules **inside** the function so `python run.py status` stays fast and tests can import `run` cheaply.
2. Add it to the `COMMANDS` dictionary.
3. If it takes a page count, add the name to the tuple in the `__main__` block that passes `max_pages`.
4. Add a line to the module docstring and a row to the command table in the README.

---

## Add a new model or listing type

**A new model for an existing type** (for example a different algorithm for
dealers): copy `training/dealer_traintest.py`, change `model_key`, point
`run_traintest` at your estimator, and add the key to `cmd_compare_models` and
`cmd_train_all` in `run.py`.

**A new listing type** (for example `motorcycle`):

1. Add the constant to `config.py` and to `VALID_LISTING_TYPES`.
2. Update the `chk_listing_type` constraint in a **new migration** (do not edit old ones).
3. Teach `classification/classify.py` to return it, with tests.
4. Add a view like the existing three, in the same migration.
5. Add a `training/<type>_traintest.py` and wire it into `run.py`.

Never let a training query run without a `listing_type` filter; this is the main
guard against datasets contaminating each other.

---

## Add a database migration

1. Create the next number: `migrations/006_short_name.sql`.
2. Start with a header comment: what changes, reversible?, modifies existing data?, expected impact.
3. Keep it idempotent (`IF NOT EXISTS`, `CREATE OR REPLACE`) and end with an insert into `schema_migrations`.
4. Put the rollback in a commented-out `DOWN` block.
5. If the change adds a column the ORM should know, also add it to `models.py` so fresh installs created by `create_tables()` match.
6. `python migrations/migrate.py --dry-run`, then apply.

Never edit a migration that others may have applied.

---

## Change the AI backend or schema

- **Different Ollama model:** just change `QWEN_MODEL` (see [ADVANCED_SETUP.md](ADVANCED_SETUP.md#ai-model-and-gpu-choices)).
- **Different provider** (an OpenAI-compatible server, a hosted API): implement a new `extract_condition(description, image_b64_list) -> ConditionAssessmentResult` with the same signature as `condition/ollama_client.py` and import it in `condition/extract.py`. Nothing else needs to change.
- **New condition field:** add it to `ConditionAssessmentResult` in `condition/schema.py` (keep it flat; nested models are less reliable with structured output), then to the mapping in `condition/merge.py`. If it should be a listing column, add a migration and a `models.py` column. Bump `CONDITION_SCHEMA_VERSION` in `config.py` so the backfill re-assesses listings under the new contract.

---

## Conventions

- **Imports and side effects.** Importing a module must not touch the network, open a browser, create folders (apart from `config.py`'s model folders) or configure global logging. `Facebook.py` was fixed for exactly this.
- **One owner per concern.** Settings in `config.py`, writes in `Ingest.py`, classification in `classification/classify.py`.
- **Log, don't print**, inside library modules. `print` is for CLI output in `run.py`.
- **Fail soft per item, loud per run.** One bad listing is logged and counted; a misconfigured environment raises immediately.
- **Tests with every behaviour change.** Bug fixes get a regression test in `tests/test_regressions.py`.
- **Keep secrets out.** `.env`, browser profiles and CSVs are ignored; add new private paths to both `.gitignore` and `.dockerignore`.
- **Commit messages** are short and say what changed and why. One logical change per commit.
