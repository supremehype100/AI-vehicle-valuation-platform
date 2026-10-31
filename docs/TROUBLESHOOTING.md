# Troubleshooting

Find your symptom, then follow the fix. If nothing here helps, run the failing
command again and read the **first** error in the output; later errors are often
side effects.

- [Setup and database](#setup-and-database)
- [Docker](#docker)
- [Scraping](#scraping)
- [AI condition extraction](#ai-condition-extraction)
- [Training and valuation](#training-and-valuation)
- [Scheduler](#scheduler)
- [Verifying a change end to end](#verifying-a-change-end-to-end)

---

## Setup and database

**`psycopg2.OperationalError: connection refused` / `could not connect to server`**
Postgres is not running or not reachable at `DATABASE_URL`. Docker: `docker compose ps` and check `db` is *healthy*. Local: start the service and check host/port. Remember that on your host the URL uses `localhost`, inside Compose it uses `db`.

**`password authentication failed for user "postgres"`**
The password in `DATABASE_URL`/`POSTGRES_PASSWORD` does not match the database. The password is fixed when the Docker volume is first created. Either use the original password, change it with `ALTER USER postgres PASSWORD '...'`, or reset with `docker compose down -v` (**deletes all data**).

**`relation "listings" does not exist`**
The tables have not been created. Run `python run.py init_db`.

**`ModuleNotFoundError: No module named 'psycopg'`**
Your `DATABASE_URL` was parsed with the wrong driver. This project rewrites a leading `postgresql://` to `postgresql+psycopg2://`. If you wrote `postgres://` (without "ql"), change it to `postgresql://`.

**`ModuleNotFoundError: No module named ...`** on any other import
You are probably outside the virtual environment, or did not run `pip install -r requirements.txt`. Activate `.venv` first.

**`Environment variable X='...' is not a valid int/float`**
A value in `.env` is not a number. Remove inline comments; `.env` does not support `VALUE # comment`.

**My `.env` changes are ignored**
`.env` must sit next to `config.py` (the project root). A real environment variable with the same name overrides the file. In Compose, the `app` service sets `DATABASE_URL` and `OLLAMA_HOST` itself.

**Migration failed halfway**
Each migration runs in one transaction, so a failure leaves nothing half-applied. Fix the cause and re-run `python migrations/migrate.py`; applied migrations are skipped.

---

## Docker

**`docker compose build app` fails to find the Dockerfile or files**
Run it from the project root. The build context is `.`

**`could not select device driver "nvidia"` / `nvidia-container-cli` errors**
Docker cannot see a GPU. Install the NVIDIA Container Toolkit, or use the CPU-only override:
`docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d db ollama`. See [GPU_SETUP.md](GPU_SETUP.md).

**Port 5432 or 11434 already in use**
Another Postgres/Ollama is running on the host. Stop it, or change the left-hand port in `docker-compose.yml` (for example `"5433:5432"`).

**The build is huge and slow**
Expected: it installs Chromium and the ML libraries. Layers are cached after the first build.

**Facebook scraping does not work in Docker**
Correct: it needs a visible browser and a logged-in profile. Run `scrape_facebook` on a desktop machine.

---

## Scraping

**0 listings returned / "Page 1 fully known"**
- If you just ran the same command, "fully known" is normal; there is nothing new.
- If the database is empty and you still get 0, the site probably changed its markup or is blocking you. Open the search URL in a browser to confirm it still works, then compare with what the parser expects (`kijiji._extract_listings_from_apollo`, `Autotrader._extract_listings`).

**`403 Forbidden` or a Cloudflare page** (most often stat.vin)
The site is challenging non-browser clients. The transport is isolated in `_fetch_page()` of that scraper; swap it for a browser-based or TLS-impersonating client. Parsing code does not need to change.

**Everything got marked `removed`**
Should not happen: the sold sweep refuses to run when it saw no listings. If a *partial* scrape ran (for example it was blocked after page 3) and you then called the sweep, listings from later pages look missing. Only run `sold_sweep` after a complete pass, and set listings back with:
```sql
UPDATE listings SET status = 'active' WHERE status = 'removed' AND last_seen > now() - interval '1 day';
```
(review the condition for your case first).

**Facebook: a browser opens but no listings are collected**
You are likely logged out or at a checkpoint. Re-do the [one-time login](ADVANCED_SETUP.md#facebook-marketplace-setup). If login is fine, Facebook changed its layout.

**Facebook: `Executable doesn't exist` (Playwright)**
Run `playwright install chromium`.

**Many listings skipped**
`normalize` drops ads with no price, a price outside $1,000-$200,000, no year or make, or "parts car" wording. That is intentional. Look at the log lines "Skipped listing ..." for the reason.

---

## AI condition extraction

**`OllamaUnavailableError` / connection refused to port 11434**
Ollama is not running or `OLLAMA_HOST` is wrong. Start it (`docker compose up -d ollama` or `ollama serve`) and check `curl $OLLAMA_HOST/api/tags`.

**`model "..." not found`**
Pull it: `ollama pull qwen3-vl:4b-thinking` (or `docker compose exec ollama ollama pull ...`). The tag must match `QWEN_MODEL` exactly.

**Timeouts, or it is extremely slow**
CPU inference is slow. Raise `OLLAMA_REQUEST_TIMEOUT_SECONDS`, lower `MAX_IMAGES_PER_LISTING` and `IMAGE_MAX_DIMENSION_PX`, or use a GPU.

**Out of memory on the GPU**
Lower `OLLAMA_NUM_CTX` and image size, or use a smaller model ([GPU_SETUP.md](GPU_SETUP.md)).

**Many `failed` assessments**
Look at `condition_assessments.error_message`. Common causes: photo URLs that expired or returned HTML, the model returning text that does not match the schema (the client retries `OLLAMA_MAX_RETRIES` times), or Ollama restarting. Failed attempts are retried on the next backfill.

**The backfill re-processes everything**
You changed `QWEN_MODEL` or the schema version. Both are part of the "already done" check by design.

---

## Training and valuation

**`Only N samples — need at least 50`**
Not enough usable rows. Scrape more (`bulk`), and check `status`. Listings without a year, with an invalid price or with parts-car wording do not count.

**`Skipped Model 4 (salvage_traintest)`**
Expected until you have at least 50 salvage lots with `auction_sale_status = 'sold'`. The stat.vin scraper currently records `pending`, so this model needs a source of final sale prices (see [MODELS.md](MODELS.md#what-the-models-use-today)).

**`Test split ... is empty` / "need at least ... training rows after split"**
After de-duplication and VIN separation too few rows remain. More data fixes it.

**`No trained model found. Run: python run.py train`**
`value` uses the Model 1 files in `MODEL_DIR`. Run `train` (or `train_private_csv`) first, from the same folder, or set `MODEL_DIR` to an absolute path. Note that a relative `MODEL_DIR` resolves against the current directory, so running from a different folder means a different (empty) `models/`.

**Predictions look wrong for a particular car**
Check how many comparable listings and training rows exist for that make and model. Rare models fall back towards the overall average. Also remember the range is ±15% heuristics, not a statistical interval.

**R² is `null` in the report**
The test set had identical prices, so R² is undefined. Stored as `null` rather than `NaN` on purpose.

**Model 1 vs Model 2 results differ a lot**
That is the point: Model 2's time-aware split is harder and more honest. See [MODELS.md](MODELS.md#why-four-models-and-two-evaluation-styles).

---

## Scheduler

**Nothing happens at 06:00 or 22:00**
The maintenance worker only acts during that exact minute and checks every 30 seconds. The scheduler must be running at that time. Check its log output.

**It scrapes too often**
The fast loop runs every 60 s between 08:00 and 22:00. Change the `60` in `fast_worker()`.

**Facebook never runs from the scheduler**
Off by default. Set `ENABLE_FACEBOOK_SCHEDULE=true` and run on a machine with a display and a logged-in profile.

**A job fails but the scheduler keeps going**
By design each job is a subprocess. Look for the traceback above the next "Running ..." line.

---

## Verifying a change end to end

The unit tests do not exercise Postgres, Playwright or Ollama. After changing code that touches them, do this smoke test against a throwaway database:

```bash
docker compose up -d db                      # or any scratch Postgres
python run.py init_db
python run.py scrape_fast                    # exercises Kijiji + AutoTrader + Ingest
python run.py status                         # counts > 0
python run.py sold_sweep                     # only after a COMPLETE pass
python -m backfill.backfill_conditions --limit 3   # needs Ollama
python run.py train                          # needs >= 50 rows
python run.py value
```

Capture the output of anything that fails and include the **first** traceback
when asking for help.
