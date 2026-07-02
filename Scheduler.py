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

        if 8 <= now.hour < 22:
            try:
                logger.info("Launching Facebook scrape (background)...")
                subprocess.Popen(
                    [PYTHON, RUN_SCRIPT, "scrape_facebook"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.exception(f"Facebook launch failed: {e}")
        else:
            logger.info("Outside hours, skipping Facebook scrape")

        # sleep 3 hours
        time.sleep(3* 60 * 60)


# -----------------------------
# START BOTH WORKERS
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting parallel scraper system...")

    t1 = threading.Thread(target=fast_worker, daemon=True)
    t2 = threading.Thread(target=facebook_worker, daemon=True)

    t1.start()
    t2.start()

    # keep main thread alive
    while True:
        time.sleep(3600)