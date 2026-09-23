# Runbook

See `docs/ARCHITECTURE.md` for the data-flow diagram and design rationale. This is the step-by-step "how do I run
X" reference.

## 1. Start Docker infrastructure

```bash
cp .env.example .env          # adjust if needed
docker compose up -d db ollama
```

This starts Postgres and Ollama only — no scraper/training/backfill job runs automatically (`docker compose up`
never triggers a job on its own, by design). If you don't have an NVIDIA GPU / the NVIDIA Container Toolkit set up,
use the CPU-only override instead:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d db ollama
```

See `docs/GPU_SETUP.md` for driver/toolkit requirements and VRAM sizing.

## 2. Start Ollama / install the Qwen model

Ollama is already running as a container from step 1. Pull the model once:

```bash
docker compose exec ollama ollama pull qwen3-vl:4b-thinking
```

(Running Ollama directly on the host instead of in Docker also works — just make sure `OLLAMA_HOST` in `.env`
points at it, and run the same `ollama pull qwen3-vl:4b-thinking`.)

## 3. Apply database migrations

```bash
python migrations/migrate.py --status      # see what's pending
python migrations/migrate.py --dry-run     # preview
python migrations/migrate.py               # apply
```

Migrations `000`-`003` and `005` are pure schema additions (safe, reversible, non-destructive). Migration `004`
(`004_backfill_listing_type.sql`) additionally **reclassifies existing rows** (sets `listing_type` from the
already-recorded `seller_type`) — read its header comment before running it against production; it's still
non-destructive (no rows deleted, fully recomputable) but it does modify existing data, so it's called out
separately per the project's migration-review convention. Run it with:

```bash
psql "$DATABASE_URL" -f migrations/004_backfill_listing_type.sql
```

## 4. Run scrapers

```bash
docker compose run --rm app python run.py scrape_fast        # Kijiji + AutoTrader, page 1
docker compose run --rm app python run.py bulk                # full bulk sweep, all sources
docker compose run --rm app python run.py scrape_statvin 5    # stat.vin salvage listings, 5 pages
docker compose run --rm app python run.py sold_sweep          # mark disappeared listings as removed
```

Facebook Marketplace (`python run.py scrape_facebook`) needs a real, already-logged-in Chrome profile
(`fb_chrome_profile_fresh/`, gitignored) and runs headed (`headless=False`) — it's not a good Docker fit (see the
caveat in `docker-compose.yml`). Run it directly on the host instead:

```bash
python run.py scrape_facebook
```

## 5. Run the historical condition backfill

```bash
docker compose run --rm app python -m backfill.backfill_conditions --resume
```

Prints an aggregate progress report (`Total records / Already processed / Remaining / Processed this run /
Successful / Failed`) — no individual listing data is ever printed. It's naturally resumable: killing it mid-run
and re-running the exact same command picks up exactly where it left off (see `backfill/backfill_conditions.py`'s
module docstring for why no separate checkpoint table is needed). Useful flags:

```bash
python -m backfill.backfill_conditions --dataset private --batch-size 100
python -m backfill.backfill_conditions --dataset salvage --limit 500
```

## 6. Train each model

```bash
docker compose run --rm app python run.py train_private_csv          # Model 1 — private / personal-CSV baseline
docker compose run --rm app python run.py train_private_traintest    # Model 2 — private / time-aware split
docker compose run --rm app python run.py train_dealer_traintest     # Model 3 — dealer / time-aware split
docker compose run --rm app python run.py train_salvage_traintest    # Model 4 — salvage / time-aware split
docker compose run --rm app python run.py train_all                  # all four, skips any without enough data
```

Model 1 needs `vehicles_cleaned.csv` (your personal validation CSV) present in the working directory to use it as
the validation set — otherwise it falls back to a random holdout, same as before this change. **This CSV is never
read, uploaded, or inspected by anyone but you** — see `scripts/evaluate_private_csv.py` below for evaluating it
without ever handing over the file itself.

## 7. Evaluate / compare models

```bash
docker compose run --rm app python run.py compare_models
```

Reads every model type's latest registered artifact from `model_artifacts` and writes `reports/model_comparison.md`
(MAE/RMSE/R²/median error/MAPE/bias/±$500/±$1k/±$2.5k/±$5k) plus prints it. For a deeper look at one model, use
`training.evaluate`'s plotting functions directly (`plot_actual_vs_predicted`, `plot_residuals`,
`plot_error_distribution`, `plot_error_by_bucket`, `plot_error_over_time`) — see `training/evaluate.py`.

To evaluate Model 1 against your personal CSV without exposing the CSV's contents to anyone:

```bash
python scripts/evaluate_private_csv.py --csv /private/path/validation.csv
```

Prints only aggregate metrics (row counts, column dtypes, missing-value %, MAE/RMSE/R²/median error/±$ thresholds)
— safe to paste back into a conversation.

## 8. Add a new model version

Every training pipeline (`training/private_csv.py`, `private_traintest.py`, `dealer_traintest.py`,
`salvage_traintest.py`) automatically registers a new, timestamped version in `model_artifacts` each time it runs
— there's no manual versioning step. `training/registry.py`'s `get_latest()` / `list_versions()` let you inspect
what's registered:

```python
from Session import get_session
from training import registry

with get_session() as session:
    print(registry.list_versions(session, "private_traintest"))
```

## 9. Resume a failed/interrupted job

- **Condition backfill**: just re-run the same `backfill.backfill_conditions` command — it's idempotent by
  construction (step 5).
- **Migrations**: re-run `python migrations/migrate.py` — already-applied migrations are tracked in
  `schema_migrations` and skipped automatically.
- **Training**: each run produces a brand-new versioned artifact; a failed run simply doesn't register one, so
  re-running is always safe (no partial-state cleanup needed).
- **Scrapers**: `scrape`/`bulk`/`scrape_fast` already skip previously-known `source_id`s via `get_known_ids()`; just
  re-run the same command.
