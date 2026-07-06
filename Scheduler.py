"""
scheduler.py

Parallel Schedule:
  Every 1 minute (8am-10pm)  — quick scrape page 1 of Kijiji + AutoTrader (Blocking, fast)
  Every 3 hours (8am-10pm)  — Facebook scrape runs in background
  Every day 6am             — full bulk maintenance sweep
  Every day 10pm            — retrain model on fresh data
"""

import sys
import time
import threading
import logging
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s"
)

logger = logging.getLogger(__name__)

PYTHON = sys.executable
RUN_SCRIPT = "run.py"


# -----------------------------
# 1. FAST LOOP (hard 60s cycle)
# -----------------------------
def fast_worker():
    while True:
        start = time.time()
        now = datetime.now()

        if 8 <= now.hour < 22:
            try:
                logger.info("Running quick scrape (Kijiji + AutoTrader)...")
                subprocess.run([PYTHON, RUN_SCRIPT, "scrape_fast"])
            except Exception as e:
                logger.exception(f"Fast scrape failed: {e}")
        else:
            logger.info("Outside operating hours, skipping fast scrape")

        # --- enforce EXACT 60s cadence (no drift) ---
        elapsed = time.time() - start
        sleep_time = max(0, 60 - elapsed)
        time.sleep(sleep_time)


# -----------------------------
# 2. FACEBOOK LOOP (3-hour cycle)
# -----------------------------
def facebook_worker():
    while True:
        now = datetime.now()

        # 1. If it's night time, sleep comfortably until 8:00 AM
        if now.hour < 8 or now.hour >= 22:
            logger.info("Outside operating hours. Facebook worker going to sleep until 8:00 AM...")

            # Calculate seconds until 8:00 AM
            target_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
            if now.hour >= 22:
                # If it's late at night, target 8am tomorrow
                from datetime import timedelta
                target_time += timedelta(days=1)

            sleep_duration = (target_time - now).total_seconds()
            time.sleep(sleep_duration)
            continue  # Wake up and re-evaluate

        # 2. If we are in bounds, fire the scraper exactly once
        try:
            logger.info("Launching Facebook scrape (background)...")
            subprocess.Popen(
                [PYTHON, RUN_SCRIPT, "scrape_facebook"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.exception(f"Facebook launch failed: {e}")

        # 3. CRITICAL: Hard sleep for 3 hours after firing to guarantee zero spamming
        logger.info("Facebook scrape dispatched. Locking loop for 3 hours...")
        time.sleep(3 * 60 * 60)

# -----------------------------
# 3. DAILY MAINTENANCE LOOP (6am & 10pm)
# -----------------------------
def maintenance_worker():
    # Flags to ensure jobs only run ONCE during their target minute
    bulk_run_today = False
    train_run_today = False
    current_day = datetime.now().date()

    while True:
        now = datetime.now()

        # Reset the daily flags if we roll over to a new day
        if now.date() != current_day:
            current_day = now.date()
            bulk_run_today = False
            train_run_today = False

        try:
            # --- 6:00 AM Bulk Sweep ---
            if now.hour == 6 and now.minute == 0 and not bulk_run_today:
                logger.info("Running global bulk scrape (including Facebook)...")
                subprocess.run([PYTHON, RUN_SCRIPT, "bulk"])
                logger.info("Running sold detection sweep...")
                subprocess.run([PYTHON, RUN_SCRIPT, "sold_sweep"])
                bulk_run_today = True

            # --- 10:00 PM Model Retraining ---
            elif now.hour == 22 and now.minute == 0 and not train_run_today:
                logger.info("Retraining model...")
                subprocess.run([PYTHON, RUN_SCRIPT, "train"])
                train_run_today = True

        except Exception as e:
            logger.exception(f"Daily maintenance job failed: {e}")

        # Check every 30 seconds so we never miss the 00-minute mark
        time.sleep(30)


# -----------------------------
# START ALL WORKERS
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting parallel scraper system...")

    t1 = threading.Thread(target=fast_worker, daemon=True)
    t2 = threading.Thread(target=facebook_worker, daemon=True)
    t3 = threading.Thread(target=maintenance_worker, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    # keep main thread alive
    while True:
        time.sleep(3600)