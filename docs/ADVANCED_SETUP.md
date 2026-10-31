# Advanced setup

For people who have the basics running ([GETTING_STARTED.md](GETTING_STARTED.md))
and want a proper, long-running or customised installation.

- [Deployment layouts](#deployment-layouts)
- [Running on a schedule](#running-on-a-schedule)
- [Changing the city or search](#changing-the-city-or-search)
- [Facebook Marketplace setup](#facebook-marketplace-setup)
- [External or managed PostgreSQL](#external-or-managed-postgresql)
- [AI model and GPU choices](#ai-model-and-gpu-choices)
- [Securing a non-local deployment](#securing-a-non-local-deployment)
- [Backups, upgrades and retention](#backups-upgrades-and-retention)
- [Tuning and capacity](#tuning-and-capacity)
- [Monitoring](#monitoring)

---

## Deployment layouts

| Layout | What runs where | When to choose it |
| --- | --- | --- |
| **Single laptop / desktop** | Everything local (Docker or bare Python) | Getting started, experimentation |
| **Compose on one server** | `db`, `ollama` and `app` containers; jobs via `docker compose run` | A home server or one VM |
| **Split workers** | Scrapers on a small always-on box; training and AI on a GPU machine; both point at one PostgreSQL | You have a GPU desktop that is not always on |
| **Facebook on a desktop** | Facebook scraping on a machine with a display and a logged-in profile; everything else elsewhere | Facebook cannot run in a headless container; see below |

All layouts share one rule: **the database is the integration point.** Any
machine that can reach the same `DATABASE_URL` can run any command. Model
files, however, live on disk in `MODEL_DIR`. If one machine trains and another
runs `value`, share that folder (network drive, rsync, a mounted volume) or run
`value` where the model was trained.

Example split: on the GPU machine

```bash
DATABASE_URL=postgresql://user:pass@db-host:5432/vehicle_intel \
OLLAMA_HOST=http://localhost:11434 \
python run.py backfill_conditions
```

---

## Running on a schedule

`Scheduler.py` is a long-running process. Start it and leave it:

```bash
python Scheduler.py
```

### The timetable (local time of the machine)

| When | What | Command it runs |
| --- | --- | --- |
| Every 60 s between 08:00 and 22:00 | Quick scrape, page 1 of Kijiji and AutoTrader | `run.py scrape_fast` |
| Every 3 h between 08:00 and 22:00 | Facebook scrape (**off by default**) | `run.py scrape_facebook` |
| Daily at 06:00 | Full sweep, in order: `bulk`, `sold_sweep`, `scrape_statvin 5`, `backfill_conditions` | one after another |
| Daily at 22:00 | Retrain every model | `run.py train_all` |

Notes:

- Each job runs in a **subprocess**, so a crash in one never kills the scheduler.
- The fast scrape starts a new run every 60 seconds measured start-to-start. That is frequent. If you would
  rather be gentler on the sites, increase the `60` in `fast_worker()` in `Scheduler.py`.
- The 06:00 job only fires if the scheduler is running during that minute. Start it before then.
- The daily sweep calls `backfill_conditions`; if Ollama is not running those listings
  simply record a failed assessment and are retried next time.
- Enable Facebook with `ENABLE_FACEBOOK_SCHEDULE=true` (needs a display, see below).

### Keeping it alive

**Linux (systemd)** `/etc/systemd/system/vehicle-intel.service`:

```ini
[Unit]
Description=Vehicle Intel scheduler
After=network-online.target postgresql.service

[Service]
WorkingDirectory=/opt/vehicle-intel
ExecStart=/opt/vehicle-intel/.venv/bin/python Scheduler.py
Restart=on-failure
RestartSec=30
User=vehicleintel
EnvironmentFile=/opt/vehicle-intel/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now vehicle-intel
journalctl -u vehicle-intel -f
```

**Windows:** use Task Scheduler ("At startup", action
`C:\path\to\.venv\Scripts\python.exe`, arguments `Scheduler.py`, start in the
project folder) or a service wrapper such as NSSM.

**Docker:** the `app` container is idle by design. To run the scheduler in a
container:

```bash
docker compose run -d --name vi-scheduler app python Scheduler.py
docker logs -f vi-scheduler
```

(Docker runs without Facebook; leave `ENABLE_FACEBOOK_SCHEDULE` unset.)

**Plain cron** also works if you prefer one job per entry, calling `run.py`
directly, e.g. `*/15 8-21 * * *  cd /opt/vehicle-intel && .venv/bin/python run.py scrape_fast`.

---

## Changing the city or search

The defaults target Calgary, Alberta.

| Source | What to change |
| --- | --- |
| Kijiji | `SEARCH_URL` in `kijiji.py`. Format `/b-cars-trucks/<city-slug>/page-{page}/c174l<location-id>`; copy the URL from a Kijiji search in your browser and keep `{page}`. `l0` means all of Canada. |
| AutoTrader | `SEARCH_URL` in `Autotrader.py`: the `reg_ab/cit_calgary` path, `zip`, `lat` and `lon` parameters. Copy the URL from a browser search and keep `page={page}`. |
| Facebook | The 12 URLs in `FB_TARGET_URLS` in `run.py`, replacing `calgary` with your city slug. |
| Facebook locale | `locale` and `timezone_id` in `Facebook._run_and_collect`. |
| Currency / bounds | `Features.is_valid_price`. |

If you change city, you will probably also want to clear or separate old data
(the fingerprint includes the city, but model training does not filter by it).

---

## Facebook Marketplace setup

Facebook is the most fragile and highest-risk source. Read this fully first.

**Risks.**
- Automated access likely violates Facebook's terms and can get an account
  restricted. Use an account you can afford to lose, keep volumes low and keep
  the `FB_*` delays at their defaults or higher.
- The browser profile *is* your login. Anyone who has `fb_chrome_profile_fresh/`
  can act as you. It is git-ignored and docker-ignored; keep it that way.

**Requirements.** A machine with a display (it opens a real Chromium window),
`playwright install chromium`, and a logged-in profile.

**One-time login.** The scraper reuses `./fb_chrome_profile_fresh`. Create the
profile and sign in once by hand:

```python
# login_once.py  (run from the project folder, then delete or ignore the file)
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context("fb_chrome_profile_fresh", headless=False)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.facebook.com/login")
    input("Log in in the browser window, then press Enter here...")
    ctx.close()
```

Then run `python run.py scrape_facebook`. Expect a browser window to open; it
visits the feed URLs in `FB_TARGET_URLS`, opens each listing and extracts the
visible text. Dealer-looking sellers are stored with `listing_type="dealer"`.

**Troubleshooting.** If you land on a login or checkpoint page, repeat the
one-time login. If you get no listings, Facebook probably changed its page
layout; the selectors live in `Facebook.extract_visible_details`.

---

## External or managed PostgreSQL

Any PostgreSQL 14+ works (RDS, Cloud SQL, Supabase, a home server).

```
DATABASE_URL=postgresql://vehicle_intel:STRONG_PASSWORD@db.example.com:5432/vehicle_intel?sslmode=require
```

1. Create an empty database and a dedicated user that owns it.
2. Point `DATABASE_URL` at it, then `python run.py init_db`.
3. Drop the `db` service (and the `depends_on` entry) from Compose, or simply never start it.
   Note that the Compose `app` service builds its own `DATABASE_URL` for the `db`
   container; for an external database, override that line in a `docker-compose.override.yml`.

The connection pool is 5 + 10 overflow (`Session.py`). Several simultaneous jobs
(scheduler workers) fit comfortably in the default `max_connections = 100`.

---

## AI model and GPU choices

| Situation | Suggestion                                                                                                |
| --- |-----------------------------------------------------------------------------------------------------------|
| NVIDIA GPU with 8-12 GB | Default `qwen3-vl:4b-thinking`, or try `qwen2.5vl:7b` (about 7-8 GB resident)                             |
| 24 GB+ GPU | Larger Qwen-VL variants may work; raise `OLLAMA_NUM_CTX` carefully                                        |
| No GPU | Dont bother                                                                                               |
| Don't want AI at all | Never run `backfill_conditions` and remove it from `Scheduler.maintenance_worker`. Everything else works. |

To swap the model:

```bash
ollama pull qwen2.5vl:7b
# in .env
QWEN_MODEL=qwen2.5vl:7b
```

Because the model name is part of each stored assessment, changing it makes the
backfill treat existing listings as unprocessed and re-assess them (the old
assessments are kept in the audit table). To avoid re-processing, do not change
`QWEN_MODEL` mid-way, or restrict a run with `--limit` and `--dataset`.

Ollama can be on another machine; set `OLLAMA_HOST=http://gpu-box:11434`.
Ollama has **no authentication**, so only expose it on a trusted network.

Full GPU instructions: [GPU_SETUP.md](GPU_SETUP.md).

---

## Securing a non-local deployment

The Compose file is a development convenience. Before exposing anything beyond localhost:

1. **Set a real `POSTGRES_PASSWORD`** in `.env` before the first `up`. (Changing it later does not change the already-initialised volume; use `ALTER USER`.)
2. **Do not publish database or Ollama ports publicly.** Change `- "5432:5432"` to `- "127.0.0.1:5432:5432"`, and likewise `11434`, or delete the `ports:` entries so only containers can reach them.
3. **Firewall** the host; allow only SSH and what you need.
4. **File permissions:** `chmod 600 .env`; keep `fb_chrome_profile_fresh/` readable only by the service user.
5. **Least privilege:** run the scheduler as a dedicated non-root user.
6. **TLS** for any remote database connection (`sslmode=require` or `verify-full`).
7. **Never commit secrets.** If one leaks, rotate it; deleting it from a later commit does not remove it from history.

---

## Backups, upgrades and retention

**Backups.** See [DATABASE.md](DATABASE.md#backups). A nightly `pg_dump` plus a
copy of `MODEL_DIR` is enough to rebuild everything.

**Upgrading to new code.**

```bash
git pull
pip install -r requirements.txt            # or: docker compose build app
python run.py init_db                      # applies any new migrations
python -m pytest -q                        # optional sanity check
```

Read any migration's header before applying it to a database you care about,
especially ones marked as modifying existing data.

**Retention.** `listing_snapshots` and `condition_assessments` only grow.
Rough scale: one snapshot per listing change. A few hundred thousand rows is
trivial for PostgreSQL. If you need to trim, delete old snapshots of removed
listings, and keep `listings` itself because training needs it.

---

## Tuning and capacity

| Goal | Knob                                                                                                  |
| --- |-------------------------------------------------------------------------------------------------------|
| Faster scraping | Lower the `random.uniform` sleeps in the scrapers (BE CAREFUL, increases the chance of being blocked) |
| Gentler scraping | Raise them, or slow the scheduler's 60 s cadence                                                      |
| Faster AI backfill | Smaller `IMAGE_MAX_DIMENSION_PX`, fewer `MAX_IMAGES_PER_LISTING`, GPU instead of CPU                  |
| Smaller GPU memory | Smaller `OLLAMA_NUM_CTX`, smaller model                                                               |
| Faster training | `n_jobs=-1` already uses all cores; fewer trees (`n_estimators`) in `Model.train` / `XGB_PARAMS`      |
| Better accuracy | More data first; then hyper-parameters                                                                |

---

## Monitoring

There is no dashboard; use logs and SQL.

- **Logs.** Every command logs to stdout/stderr with timestamps. Under systemd use `journalctl`; under Docker use `docker logs`.
- **Is data flowing?**
  ```sql
  SELECT source, max(last_seen) AS last_activity, count(*) FILTER (WHERE status='active') AS active
  FROM listings GROUP BY source;
  ```
  A source whose `last_activity` stops advancing has probably broken or been blocked.
- **Did the nightly training run?**
  ```sql
  SELECT model_key, max(trained_at) FROM model_artifacts GROUP BY model_key;
  ```
- **Is the AI backfill progressing?** Run it with `--limit 1` and read the aggregate report, or count `condition_assessments` by `extraction_status`.
- **Scraper breakage** is the most common failure. Websites change their markup. The usual symptom is "0 listings". If a full scrape sees nothing, the sold sweep logs a warning and skips instead of marking everything as removed.
