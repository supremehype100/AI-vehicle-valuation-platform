"""
scheduler.py

Parallel Schedule:
  Every 1 minute (8am-10pm)  — quick scrape page 1 of Kijiji + AutoTrader (Blocking, fast)
  Every 3 hours (8am-10pm)  — Facebook scrape runs in background (Non-blocking via Popen)
  Every day 6am             — full bulk maintenance sweep
  Every day 10pm            — retrain model on fresh data
"""

import schedule
import sys
import time
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

# Track the running Facebook background process object
fb_process = None


def scrape_new():
    now = datetime.now()
    if now.hour == 6 and now.minute == 0:
        return  # Let bulk take over at 6am

    if 8 <= now.hour < 22:
        logger.info("Running quick scrape (Kijiji + AutoTrader)...")
        # Keeps this blocking because it only takes 5-10 seconds
        subprocess.run([PYTHON, RUN_SCRIPT, "scrape_fast"])
    else:
        logger.info("Outside 8am-10pm window, skipping quick scrape")


def scrape_facebook():
    global fb_process
    now = datetime.now()
    if now.hour == 6 and now.minute == 0:
        return

    if 8 <= now.hour < 22:
        # ─── PROCESS GUARD CHECK ───────────────────────────────────────
        # Check if a previous Facebook scrape is still active in the background
        if fb_process is not None and fb_process.poll() is None:
            logger.warning(
                " Facebook scraper is STILL running from the last cycle! Skipping spawning to prevent collision.")
            return

        logger.info(" Launching Facebook scrape in the BACKGROUND...")
        # Popen runs it asynchronously, freeing the scheduler to handle 1-min intervals
        fb_process = subprocess.Popen([PYTHON, RUN_SCRIPT, "scrape_facebook"])
    else:
        logger.info("Outside operating hours, skipping Facebook scrape")


def scrape_full():
    logger.info("Running global bulk scrape (including Facebook)...")
    subprocess.run([PYTHON, RUN_SCRIPT, "bulk"])
    logger.info("Running sold detection sweep...")
    subprocess.run([PYTHON, RUN_SCRIPT, "sold_sweep"])


def train():
    logger.info("Retraining model...")
    subprocess.run([PYTHON, RUN_SCRIPT, "train"])


# --- Hooking up the schedules ---
schedule.every(1).minutes.do(scrape_new)
schedule.every(3).hours.do(scrape_facebook)
schedule.every().day.at("06:00").do(scrape_full)
schedule.every().day.at("22:00").do(train)

logger.info("Scheduler started successfully in Parallel Mode")

while True:
    schedule.run_pending()
    time.sleep(1)