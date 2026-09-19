"""
scheduler.py

Parallel Schedule:
  Every 1 minute (8am-10pm) — quick scrape page 1 of Kijiji + AutoTrader (blocking, fast)
  Every 3 hours (8am-10pm)  — Facebook scrape runs in background
  Every day 6:00 AM         — full bulk maintenance sweep:
                                 bulk scrape -> sold sweep -> stat.vin salvage scrape
                                 -> AI condition backfill (GPU/Ollama, runs once here only)
  Every day 10:00 PM        — retrain ALL FOUR models (private_csv, private_traintest,
                                 dealer_traintest, salvage_traintest) via train_all
"""

import sys
import time
import threading
import logging
import subprocess
from datetime import datetime, timedelta

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
#def facebook_worker():
    while True:
        now = datetime.now()

        if now.hour < 8 or now.hour >= 22:
            logger.info("Outside operating hours. Facebook worker going to sleep until 8:00 AM...")
            target_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
            if now.hour >= 22:
                target_time += timedelta(days=1)
            sleep_duration = (target_time - now).total_seconds()
            time.sleep(sleep_duration)
            continue

        try:
            logger.info("Launching Facebook scrape (background)...")
            subprocess.Popen(
                [PYTHON, RUN_SCRIPT, "scrape_facebook"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.exception(f"Facebook launch failed: {e}")

        logger.info("Facebook scrape dispatched. Locking loop for 3 hours...")
        time.sleep(3 * 60 * 60)


# ------------------------------------------------------------------
# 3. DAILY MAINTENANCE LOOP
#
#    6:00 AM  — bulk scrape -> sold sweep -> stat.vin salvage scrape
#               -> AI condition backfill (GPU/Ollama work)
#
#    Backfill is chained directly after scraping (not scheduled at its
#    own fixed clock-minute) because subprocess.run() blocks this thread
#    while bulk/sold_sweep/statvin run. If backfill instead waited for
#    an exact minute like 6:30, and bulk was still running past 6:30,
#    the loop would never get back to the top in time to catch that
#    exact minute and would skip a full day. Chaining sidesteps that.
#
#    10:00 PM — retrain ALL FOUR models via train_all
# ------------------------------------------------------------------
def maintenance_worker():
    bulk_run_today = False
    train_run_today = False
    current_day = datetime.now().date()

    while True:
        now = datetime.now()

        if now.date() != current_day:
            current_day = now.date()
            bulk_run_today = False
            train_run_today = False

        try:
            # --- 6:00 AM: scrape everything, then backfill conditions ---
            if now.hour == 6 and now.minute == 0 and not bulk_run_today:
                logger.info("Running global bulk scrape (including Facebook)...")
                subprocess.run([PYTHON, RUN_SCRIPT, "bulk"])

                logger.info("Running sold detection sweep...")
                subprocess.run([PYTHON, RUN_SCRIPT, "sold_sweep"])

                logger.info("Scraping stat.vin salvage listings...")
                subprocess.run([PYTHON, RUN_SCRIPT, "scrape_statvin", "5"])

                logger.info("Running AI condition backfill (GPU machine)...")
                subprocess.run([PYTHON, RUN_SCRIPT, "backfill_conditions"])

                bulk_run_today = True

            # --- 10:00 PM: retrain all four models ---
            elif now.hour == 22 and now.minute == 0 and not train_run_today:
                logger.info(
                    "Retraining all models (private_csv, private_traintest, "
                    "dealer_traintest, salvage_traintest)..."
                )
                subprocess.run([PYTHON, RUN_SCRIPT, "train_all"])
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
    #t2 = threading.Thread(target=facebook_worker, daemon=True)
    t3 = threading.Thread(target=maintenance_worker, daemon=True)

    t1.start()
    #t2.start()
    t3.start()

    # keep main thread alive
    while True:
        time.sleep(3600)