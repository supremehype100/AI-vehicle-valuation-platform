# syntax=docker/dockerfile:1
#
# Single image used for BOTH the scrapers (Kijiji/AutoTrader/Facebook/stat.vin)
# and the training/backfill jobs (XGBoost training, condition backfill). They
# share one requirements.txt, so there is no benefit to splitting into separate
# images — just run different commands against this one image (see
# docker-compose.yml's top-of-file comment block for the exact invocations).
#
# This image is intentionally large: Playwright's Chromium + its OS-level deps
# alone are several hundred MB, on top of the ML stack (xgboost, scikit-learn,
# pandas, numpy, Pillow, matplotlib). That's expected for a scraper+ML image —
# do not "optimize" this away by dropping --with-deps or slimming the base
# unless you're prepared to debug missing shared libraries at Playwright
# runtime.
FROM python:3.12-slim

# Playwright needs a real (non-headless-hostile) locale/timezone base and a
# handful of system libs before `playwright install --with-deps` will even run;
# psycopg2-binary and lxml also want build/runtime system libs. Keep apt lists
# out of the final layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# `playwright install --with-deps chromium` pulls Chromium plus every OS
# package Chromium needs (fonts, codecs, X11 libs, etc.) — this is what makes
# the image large, and it's the price of Facebook/Kijiji/AutoTrader scraping
# working reliably in a minimal base image instead of hand-maintaining that
# dependency list ourselves.
RUN playwright install --with-deps chromium

# Now copy the actual project code. .dockerignore keeps fb_chrome_profile_fresh/,
# fb_session/, *.csv, models/*.pkl, image_cache/, reports/, and .git out of the
# build context entirely.
COPY . .

# Directories config.py expects to exist / write into at runtime — created here
# so a fresh container has them even before any volume is mounted; in
# docker-compose.yml these are overlaid by bind mounts so the real contents
# live on the host and survive container recreation.
RUN mkdir -p /app/models /app/image_cache

# No ENTRYPOINT/CMD baked in beyond a harmless default — docker-compose.yml's
# `app` service overrides this with an idle no-op so `docker compose up` never
# accidentally starts a scrape/train/backfill job. Real work is invoked with
# `docker compose run --rm app python run.py <command>` etc.
CMD ["python", "-c", "print('app image ready — use: docker compose run --rm app <command>')"]
