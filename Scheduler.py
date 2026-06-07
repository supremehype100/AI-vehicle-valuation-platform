"""
scheduler.py

Schedule:
  Every 1 minute  (8am-10pm) — scrape page 1 of Kijiji + AutoTrader for new listings
  Every day 6am              — full scrape all pages + sold detection sweep
  Every day 10pm             — retrain model on fresh data
"""

import schedule
import time
import logging
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def scrape_new():
    """Quick scrape — page 1 only, new listings only."""
    now = datetime.now()
    if 8 <= now.hour < 22:
        logger.info("Running quick scrape (page 1)...")
        subprocess.run(["python", "run.py", "scrape", "1"])
    else:
        logger.info("Outside 8am-10pm window, skipping quick scrape")


def scrape_full():
    """Full scrape all pages + sold detection."""
    logger.info("Running full scrape + sold detection sweep...")
    subprocess.run(["python", "run.py", "bulk"])
    subprocess.run(["python", "run.py", "sold_sweep"])


def train():
    logger.info("Retraining model...")
    subprocess.run(["python", "run.py", "train"])


schedule.every(1).minutes.do(scrape_new)
schedule.every().day.at("06:00").do(scrape_full)
schedule.every().day.at("22:00").do(train)

logger.info("Scheduler started")
logger.info("  Every minute (8am-10pm): quick scrape page 1")
logger.info("  Daily 6am: full scrape + sold detection")
logger.info("  Daily 10pm: retrain model")

while True:
    schedule.run_pending()
    time.sleep(10)