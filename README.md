# Vehicle Intel

**A self-hosted pipeline that collects used-vehicle listings, learns what cars
are really selling for, and tells you what a given car is worth.**

It scrapes Kijiji, AutoTrader, Facebook Marketplace and stat.vin salvage
auctions, keeps a full price and status history in PostgreSQL, optionally reads
listing text and photos with a local vision-language AI model to judge
condition, and trains XGBoost price models on the result. Everything runs on
your own hardware; there are no paid APIs.

> New here? Read [**What this project does, in plain English**](docs/OVERVIEW.md),
> then follow [**Getting started**](docs/GETTING_STARTED.md).

---

## Features

- **Four listing sources** behind one common interface: Kijiji, AutoTrader, Facebook Marketplace, stat.vin.
- **History, not snapshots:** every price change is recorded; vanished listings are marked removed with their days on market.
- **Private / dealer / salvage classification**, so different markets never contaminate each other.
- **Duplicate detection** with VINs and a vehicle fingerprint.
- **Local AI condition reading** (Ollama + a Qwen vision model): structured ratings and red-flag indicators with an audit trail and a rule that AI never overwrites trusted data.
- **Four price models** with a leakage-aware evaluation (time-based split, VIN-disjoint, de-duplicated) and a versioned model registry.
- **Valuation:** predicted price, likely range and comparable active listings.
- **Scheduler** for unattended collection, nightly sweeps and retraining.
- **Docker Compose stack** (Postgres, Ollama, app) with GPU and CPU-only modes.
- **127 tests** that need no database, network or GPU.

## How it works

```
 Kijiji ─┐
 AutoTrader ─┤   normalize    ┌──────────┐   classify,    ┌────────────┐
 Facebook ─┼────────────────► │  Ingest  │ ─ dedupe, ───► │ PostgreSQL │
 stat.vin ─┘                  └──────────┘   track changes└─────┬──────┘
                                                               │
                         ┌─────────────────────────────────────┼───────────────┐
                         ▼                                     ▼               ▼
                 AI condition reading                  Feature building   Sold sweep
              (Ollama, text + photos, optional)               │          (marks removals)
                                                              ▼
                                            XGBoost models 1-4 + registry
                                                              │
                                                              ▼
                                       value: price, range, comparable listings
```

## Quick start

### With Docker (recommended)

```bash
git clone <your-repo-url> vehicle-intel && cd vehicle-intel
cp .env.example .env

docker compose up -d db ollama                     # database + AI server
docker compose exec ollama ollama pull qwen3-vl:4b-thinking   # optional, for AI condition reading
docker compose build app
docker compose run --rm app python run.py init_db  # create the tables
docker compose run --rm app python run.py scrape_fast
docker compose run --rm app python run.py status
```

No NVIDIA GPU? Add `-f docker-compose.yml -f docker-compose.cpu-only.yml` to the `up` command.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium                          # only for Facebook
cp .env.example .env                                 # point DATABASE_URL at your Postgres
python run.py init_db
python run.py scrape_fast
python run.py status
```

Once you have 50+ listings: `python run.py train`, then `python run.py value`.
The full walk-through, with expected output, is in [Getting started](docs/GETTING_STARTED.md).

## Command cheat sheet

Run as `python run.py <command>` (or `docker compose run --rm app python run.py <command>`).

| Command | What it does |
| --- | --- |
| `init_db` | Create tables and apply SQL migrations (run once, safe to repeat) |
| `status` | Row counts by source and make |
| `scrape_fast [pages]` | Page 1 of Kijiji + AutoTrader |
| `scrape [pages]` | Kijiji + AutoTrader, default 5 pages |
| `bulk [pages]` | Every page of both, ignoring known ids |
| `scrape_statvin [pages]` | stat.vin salvage lots |
| `scrape_facebook` | Facebook Marketplace (needs a logged-in browser profile) |
| `sold_sweep` | Mark listings that disappeared as removed |
| `backfill_conditions` | AI condition extraction over the database (needs Ollama) |
| `train` | Baseline price model |
| `train_all` | All four models; skips any without enough data |
| `compare_models` | Write `reports/model_comparison.md` |
| `compare_private_fair` | Score models 1 and 2 on the same held-out rows |
| `value` | Interactive "what is this car worth?" |
| `pipeline [pages]` | Scrape, then train |
| `python Scheduler.py` | Run the whole timetable unattended |

## Documentation

| If you want to... | Read |
| --- | --- |
| Understand what this is, with no jargon | [docs/OVERVIEW.md](docs/OVERVIEW.md) |
| Install and run it step by step | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| See every setting | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| Understand each component and its interface | [docs/MODULES.md](docs/MODULES.md) |
| Design: data flow and the reasoning behind it | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Know the tables, migrations and useful SQL | [docs/DATABASE.md](docs/DATABASE.md) |
| Understand how prices are predicted and evaluated | [docs/MODELS.md](docs/MODELS.md) |
| Deploy for real: scheduling, Facebook, security, backups | [docs/ADVANCED_SETUP.md](docs/ADVANCED_SETUP.md) |
| Operate it day to day | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Set up a GPU or CPU-only AI server | [docs/GPU_SETUP.md](docs/GPU_SETUP.md) |
| Add a source, feature, model or command | [docs/EXTENDING.md](docs/EXTENDING.md) |
| Fix a problem | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

## Project layout

```
run.py                  command-line entry point
Scheduler.py            unattended timetable (calls run.py)
config.py               all settings, read from environment / .env
models.py               database tables and data shapes
Session.py              database connection helper
Ingest.py               the only writer of listings: upsert, dedupe, sold sweep
Features.py             listings -> model inputs
Model.py                baseline model, training and valuation
kijiji.py  Autotrader.py  Facebook.py  scrapers/statvin.py     data sources
classification/         private / dealer / salvage decision
condition/              AI condition extraction (schema, images, Ollama client, merge rules)
backfill/               run condition extraction over the database
training/               splits, leakage guard, metrics, registry, the four pipelines, evaluation
migrations/             numbered SQL files + runner
scripts/                helper utilities
tests/                  127 tests (no DB / network / GPU needed)
docs/                   documentation
Dockerfile  docker-compose*.yml  .env.example
```

## Requirements

| Needed | For |
| --- | --- |
| Python 3.12 | everything (Docker image uses 3.12) |
| PostgreSQL 14+ | storage (16 in Compose) |
| Docker + Compose v2 | the recommended route |
| Ollama + a vision model | AI condition reading only (optional) |
| NVIDIA GPU, 8 GB+ VRAM | fast AI reading (optional; CPU works, slowly) |
| Chromium via Playwright | Facebook only |

## Status and known limitations

- The default searches target **Calgary, Alberta**. See [changing the city](docs/ADVANCED_SETUP.md#changing-the-city-or-search).
- Prices are **asking prices**, not final sale prices.
- AI condition data is stored and queryable but is **not yet a model input**.
- The **stat.vin** scraper has not been verified end to end against the live site, and it records bids rather than final sale prices, so the salvage model (Model 4) has no training data until sold results are captured.
- The Facebook scraper needs a display and a logged-in profile; it is the most fragile source.
- Dealer detection is keyword based and will occasionally misclassify an ad.
- Scrapers depend on third-party page layouts and will need maintenance when those change.
- The valuation range is a heuristic (±15%, tightened by comparables), not a statistical interval.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Responsible use

Check each website's terms of service and `robots.txt` before scraping. Keep
request volume low, keep the built-in delays, do not republish scraped content
and use the Facebook scraper only with your own account (automated access may
breach its terms and can lead to account restrictions). This project is
provided for personal research and learning. You are responsible for how you
use it.

## Security

Never commit `.env`, `fb_chrome_profile_fresh/`, `fb_session/` or any CSV of
private data; they are git-ignored for a reason. The Facebook profile is
effectively your login. See the [security checklist](docs/CONFIGURATION.md#security-checklist)
and [hardening notes](docs/ADVANCED_SETUP.md#securing-a-non-local-deployment).

## License

No license has been chosen yet. Until one is added, all rights are reserved by
the author. Add a `LICENSE` file (for example MIT or Apache-2.0) if you want
others to be able to use the code.
