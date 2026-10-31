# Getting started

This guide takes you from a fresh download to your first price prediction. It
covers two routes; pick one.

| Route | Best for | You install yourself |
| --- | --- | --- |
| **A. Docker** (recommended) | Most people; least can go wrong | Docker Desktop (or Docker Engine + Compose v2), Git |
| **B. Run on your machine** | Developers, or when Docker is not an option | Python 3.12, PostgreSQL 14+, Git (and optionally Ollama) |

Not sure what the pieces are? Read [OVERVIEW.md](OVERVIEW.md) first.

**Time needed:** about 20 minutes, plus however long you let the scrapers run.
You need roughly **a few hundred listings** before price models are worth
looking at, and a minimum of **50** before training will run at all.

---

## 0. Get the code

```bash
git clone <your-repo-url> vehicle-intel
cd vehicle-intel
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

`.env` holds your settings. The defaults work for a local setup, so you only
need to edit it if you want a different database password or city. Every
setting is listed in [CONFIGURATION.md](CONFIGURATION.md). **Never commit
`.env`.** It is already git-ignored.

---

## Route A: Docker

### A1. Start the database and AI server

```bash
docker compose up -d db ollama
```

- `db` is PostgreSQL 16 (data lives in a Docker volume called `db_data`).
- `ollama` is the local AI server (models live in the `ollama_data` volume).
- This does **not** start any scraping or training. Those are explicit,
  one-off commands you run yourself.

**No NVIDIA GPU?** Use the CPU-only override. It works but AI condition
extraction will be slow:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d db ollama
```

GPU details are in [GPU_SETUP.md](GPU_SETUP.md).

### A2. Download the AI model (one time, a few GB)

```bash
docker compose exec ollama ollama pull qwen3-vl:4b-thinking
```

You can skip this if you only want scraping and price models. The AI condition
step is optional.

### A3. Build the app image and create the tables

```bash
docker compose build app
docker compose run --rm app python run.py init_db
docker compose run --rm app python run.py status
```

The first build is slow (it downloads Chromium and the ML libraries). `init_db`
creates every table and applies the SQL migrations. `status` should print
`Total listings: 0`.

### A4. Collect data

```bash
# page 1 of Kijiji and AutoTrader: quick, a minute or two
docker compose run --rm app python run.py scrape_fast

# every page of both sites: much more data, can take a long while
docker compose run --rm app python run.py bulk

docker compose run --rm app python run.py status
```

Re-running is always safe: listings are matched on (source, id), so you get
updates, never duplicates.

### A5. Train and value

```bash
docker compose run --rm app python run.py train
docker compose run --rm app python run.py value
```

`value` is interactive. It asks for make, model, year and mileage and prints
the predicted price, a likely range and comparable active listings.

Skip ahead to [What you should see](#what-you-should-see).

---

## Route B: run on your machine

### B1. Install the prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.12 (the Docker image uses 3.12) | 3.10+ is likely fine but untested |
| PostgreSQL | 14 or newer (16 used in Compose) | Any install method works |
| Git | any | |
| Ollama | latest | Optional, only for AI condition extraction |

### B2. Create a database

```bash
# macOS / Linux
createdb vehicle_intel
# or in psql:  CREATE DATABASE vehicle_intel;
```

If your Postgres user or password differs from `postgres`/`postgres`, set
`DATABASE_URL` in `.env`, for example:

```
DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/vehicle_intel
```

### B3. Create a virtual environment and install packages

```bash
python -m venv .venv

# activate it:
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt
playwright install chromium        # only needed for the Facebook scraper
```

### B4. Create the tables and check the connection

```bash
python run.py init_db
python run.py status
```

### B5. (Optional) Set up the AI model

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull qwen3-vl:4b-thinking
```

The default `OLLAMA_HOST=http://localhost:11434` already points at it.

### B6. Collect data, train, value

```bash
python run.py scrape_fast
python run.py bulk
python run.py status
python run.py train
python run.py value
```

---

## What you should see

After `init_db`:

```
Database ready.
```

After `status` with data in the database (numbers will differ):

```
=== DB Status ===
  Total listings:     3,412
  With price:         3,380
  With make/model:    3,401
  Snapshots:          3,412

By source:
  kijiji                 2105
  autotrader             1307
```

After `train`, a model summary with the number of training samples, the mean
absolute error in dollars and the R² score. Files are written to `models/`.

After `value`, something like:

```
  Predicted price:  $    18,450
  Likely range:     $    16,200  –  $20,100
  Model MAE:        $     1,830
  Trained on:       2840 listings
```

---

## Going further

| I want to... | Command / document |
| --- | --- |
| Read photos and text with AI to judge condition | `python run.py backfill_conditions` (needs Ollama), see [RUNBOOK.md](RUNBOOK.md#5-run-the-historical-condition-backfill) |
| Scrape salvage-auction lots | `python run.py scrape_statvin 5` (read the caveats in [MODULES.md](MODULES.md#statvin-scraper)) |
| Scrape Facebook Marketplace | Needs a logged-in browser profile: [ADVANCED_SETUP.md](ADVANCED_SETUP.md#facebook-marketplace-setup) |
| Train all four models and compare them | `python run.py train_all` then `python run.py compare_models` |
| Run everything on a schedule | `python Scheduler.py`, see [ADVANCED_SETUP.md](ADVANCED_SETUP.md#running-on-a-schedule) |
| Use a different city | [ADVANCED_SETUP.md](ADVANCED_SETUP.md#changing-the-city-or-search) |
| Change a setting | [CONFIGURATION.md](CONFIGURATION.md) |
| Something broke | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

## Quick sanity checklist

- [ ] `docker compose ps` (Route A) shows `db` as *healthy*.
- [ ] `python run.py status` connects without an error.
- [ ] After `scrape_fast`, `Total listings` is greater than 0.
- [ ] You have at least 50 listings with a price, year and make before `train`.
- [ ] `.env` is not tracked: `git status` must not list it.
