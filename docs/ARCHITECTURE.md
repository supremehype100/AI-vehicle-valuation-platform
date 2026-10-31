# Architecture

This page records the data flow and *why* the system is shaped the way it is.
For a friendlier introduction see [OVERVIEW.md](OVERVIEW.md); for what each
module does see [MODULES.md](MODULES.md).

Note: the condition fields extracted by the AI step are stored on each listing
but are not yet consumed by the feature-engineering step (the arrow below shows
where they would plug in).

## Data flow

```
Scrapers (Kijiji, AutoTrader, Facebook, stat.vin)
   |
   v
Raw listing dicts (Ingest.upsert)
   |
   v
Classification (classification/classify.py -> Listing.listing_type)
   |
   +---------------+---------------+
   v               v               v
Private table   Dealer table   Salvage table      (logical split via
(listing_type=  (listing_type= (listing_type=       listing_type column
 'private')      'dealer')      'salvage')          + private/dealer/
   |               |               |                salvage_listings views)
   +---------------+---------------+
                   |
                   v
       Condition extraction (condition/ + backfill/)
         - Text:   Listing.description
         - Images: Listing.image_urls -> Ollama / Qwen vision-language model
         - Output: condition_assessments (audit trail, insert-only)
                   -> merged into Listing.condition_* (precedence-ruled)
                   |
                   v
       Feature engineering (Features.py, training/leakage.py)
                   |
                   v
       Training (training/ + Model.py)
         +-- Model 1: Private / existing personal-CSV validation  (training/private_csv.py, wraps Model.py unchanged)
         +-- Model 2: Private / new time-aware train-test split   (training/private_traintest.py)
         +-- Model 3: Dealer  / new time-aware train-test split   (training/dealer_traintest.py)
         +-- Model 4: Salvage / new time-aware train-test split   (training/salvage_traintest.py)
                   |
                   v
       Model registry (model_artifacts table, training/registry.py)
                   |
                   v
       Evaluation / comparison (training/evaluate.py)
         - metrics table (reports/model_comparison.md)
         - plots (reports/*.png)
```

## Why a single Postgres table with a `listing_type` discriminator, not three databases

The design calls for three separate datasets (private/dealer/salvage) that must never contaminate each other. Three fully
separate Postgres databases (or schemas) would mean triplicating `listings`, `listing_snapshots`, `price_predictions`,
and `condition_assessments`, plus every foreign key and every query in `Ingest.py`/`Features.py`. Given the existing
architecture is one Postgres database with one `listings` table, the lower-risk, equally-safe design is:

- An explicit, indexed, `NOT NULL` `listing_type` column (`'private' | 'dealer' | 'salvage'`), enforced by a `CHECK`
  constraint — not a boolean, not an inferred value.
- Every training data loader (`Features.load_training_data`, and every `training/*_traintest.py` pipeline) takes and
  requires a `listing_type` filter — there is no code path that reads "all listings" without the caller explicitly
  asking for that.
- Three read-only convenience views (`private_listings`, `dealer_listings`, `salvage_listings`) so ad hoc queries have
  an unambiguous, self-documenting table name to reach for instead of remembering to add `WHERE listing_type = ...`
  by hand.
- `classification/classify.py` is the single place that decides a listing's type — every scraper's output is
  classified consistently, and `Ingest.upsert()` calls it automatically at insert time.

This gives the same contamination guarantees (a query for the private dataset structurally cannot see salvage rows
unless it explicitly asks to) without tripling the schema surface area.

## Audit-trail vs. current-state pattern (reused, not invented)

The existing codebase already has this pattern: `Listing` holds the current state of a listing, `listing_snapshots`
is an insert-only history of every observed state (used to detect sales/removals). The condition-extraction system
reuses the same shape:

- `condition_assessments` — insert-only, one row per AI extraction attempt (successful or failed), with the images
  used, a hash of the description text at extraction time, the full structured model output, and confidence.
- `Listing.condition_*` — the current, precedence-merged view (what the feature pipeline and app actually read).

## Precedence merge (backfill correctness)

`condition/merge.py` implements: manually validated (40) > trusted structured data (30) > AI-extracted (20) >
inferred/weak (10). A field's current precedence is stored per-field in `Listing.condition_field_provenance`
(JSONB). A new value only overwrites an existing one if its precedence is `>=` the stored precedence — this is what
makes it structurally impossible for a backfill re-run to clobber a manually-validated field, and is covered by
`tests/test_condition_merge.py`.

## Known pre-existing leakage issue in the baseline (Model 1)

`Features.listings_to_dataframe()` includes `days_on_market` as a training feature, using each listing's *current*
value. For sold/removed listings that reflects information only knowable after the fact, while `Model.valuate()`
always predicts with `days_on_market=0`. This is a real, pre-existing leakage bug inherited from before this change,
left untouched in `Model.py` (Model 1 must stay reproducible), but documented in `training/leakage.py` and
explicitly neutralized (forced to 0 for all rows) in every new train-test pipeline (Models 2-4).

## GPU / AI pipeline

`condition/ollama_client.py` talks to a local Ollama server (`OLLAMA_HOST`) running a quantized Qwen vision-language model
(`QWEN_MODEL`, default `qwen3-vl:4b-thinking`) over its REST API — no separate `ollama` Python package, reusing the
project's existing `httpx` dependency. See `docs/GPU_SETUP.md` (or the top of `docker-compose.yml`) for VRAM sizing
against the target GTX 1080 Ti (11GB).
