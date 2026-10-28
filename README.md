# Vehicle Intel

Scrapes used-vehicle listings (Kijiji, AutoTrader, Facebook Marketplace and
stat.vin salvage auctions), stores them in Postgres with price/status history,
extracts condition from listing text and photos with a local vision-language
model, and trains XGBoost price models on the result.

## Quick start

```bash
cp .env.example .env                       # adjust values if needed
docker compose up -d db ollama
docker compose exec ollama ollama pull qwen3-vl:4b-thinking
docker compose run --rm app python run.py status
```

Running directly on your host instead:

```bash
pip install -r requirements.txt
playwright install chromium
python run.py status
```

## Common commands

| Command | What it does |
| --- | --- |
| `python run.py scrape_fast` | Page 1 of Kijiji + AutoTrader |
| `python run.py bulk [pages]` | Every page of every source, ignoring known ids |
| `python run.py scrape_statvin [pages]` | stat.vin salvage listings |
| `python run.py scrape_facebook` | Facebook Marketplace (needs a logged-in profile) |
| `python run.py sold_sweep` | Mark listings that disappeared as removed |
| `python run.py backfill_conditions` | AI condition extraction over the DB |
| `python run.py train_all` | Train all four price models |
| `python run.py compare_models` | Write `reports/model_comparison.md` |
| `python Scheduler.py` | Run the scrape/maintenance/retrain schedule |

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for operations,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit together
and [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) for GPU/CPU Ollama setup.

## Configuration

Everything is read from environment variables (or a `.env` file next to
`config.py`); see [`.env.example`](.env.example) for the full list and
defaults. Never commit `.env`, the Facebook browser profile
(`fb_chrome_profile_fresh/`, `fb_session/`) or any CSV of private data; they
are all git-ignored.

## Tests

```bash
pip install -r requirements.txt
python -m pytest
```

The suite needs no database or network access.

## Scraping notice

Check each site's terms of service and `robots.txt` before scraping, keep
request volume low, and only use the Facebook scraper with your own account.
