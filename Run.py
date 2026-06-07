"""
run.py — Main entry point.

Commands:
    python run.py scrape        — scrape Kijiji and store to DB
    python run.py train         — train the price model on stored data
    python run.py value         — interactive valuation prompt
    python run.py pipeline      — scrape + train in one shot
    python run.py status        — show DB stats
"""

from __future__ import annotations
import sys
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_scrape(max_pages: int = 5, bulk: bool = False):
    from Session import create_tables, get_session
    from Ingest import upsert_batch, get_known_ids
    from kijiji import run as kijiji_run
    from Autotrader import run as autotrader_run

    create_tables()

    with get_session() as session:
        known_kijiji = set() if bulk else get_known_ids("kijiji", session)
        known_autotrader = set() if bulk else get_known_ids("autotrader", session)

    # Kijiji
    kijiji_listings = list(kijiji_run(max_pages=max_pages, known_ids=known_kijiji))
    logger.info("Scraped %d listings from Kijiji", len(kijiji_listings))
    with get_session() as session:
        kc = upsert_batch(kijiji_listings, session)
    print(f"\nKijiji    — New: {kc['new']} | Updated: {kc['updated']} | Errors: {kc['errors']}")

    # AutoTrader
    at_listings = list(autotrader_run(max_pages=max_pages, known_ids=known_autotrader))
    logger.info("Scraped %d listings from AutoTrader", len(at_listings))
    with get_session() as session:
        ac = upsert_batch(at_listings, session)
    print(f"AutoTrader — New: {ac['new']} | Updated: {ac['updated']} | Errors: {ac['errors']}")


def cmd_bulk():
    cmd_scrape(max_pages=100, bulk=True)


def cmd_sold_sweep():
    from Session import get_session
    from Ingest import sold_sweep, get_known_ids
    from kijiji import run as kijiji_run
    from Autotrader import run as autotrader_run

    logger.info("Running sold sweep...")

    # Scrape all pages to get current live IDs
    kijiji_seen = {l["source_id"] for l in kijiji_run(max_pages=100)}
    at_seen = {l["source_id"] for l in autotrader_run(max_pages=100)}

    with get_session() as session:
        kc = sold_sweep("kijiji", kijiji_seen, session)
        ac = sold_sweep("autotrader", at_seen, session)

    print(f"\nSold sweep complete:")
    print(f"  Kijiji removed:    {kc['removed']}")
    print(f"  AutoTrader removed: {ac['removed']}")


def cmd_train():
    from Session import get_session
    from Model import train

    with get_session() as session:
        try:
            meta = train(session)
        except ValueError as e:
            print(f"\n{e}")
            return

    print(f"\nModel trained:")
    print(f"  Samples:       {meta['training_samples']}")
    print(f"  MAE:           ${meta['mae']:,.0f}")
    print(f"  R²:            {meta['r2']}")
    print(f"  Median error:  {meta['median_pct_error']}%")


def cmd_value():
    from Session import get_session
    from Model import valuate

    print("\n=== Vehicle Valuation ===")
    make = input("Make (e.g. toyota): ").strip()
    model_name = input("Model (e.g. camry): ").strip()
    year = int(input("Year (e.g. 2019): ").strip())
    mileage = int(input("Mileage in km (e.g. 80000): ").strip())
    trim = input("Trim (optional, press Enter to skip): ").strip() or None

    with get_session() as session:
        try:
            result = valuate(
                session,
                make=make,
                model_name=model_name,
                year=year,
                mileage=mileage,
                trim=trim,
            )
        except FileNotFoundError:
            print("\nNo trained model found. Run: python run.py train")
            return

    print(f"\n{'─'*40}")
    print(f"  Predicted price:  ${result['predicted_price']:>10,.0f}")
    print(f"  Likely range:     ${result['confidence_low']:>10,.0f}  –  ${result['confidence_high']:,.0f}")
    print(f"  Model MAE:        ${result['model_mae']:>10,.0f}")
    print(f"  Trained on:       {result['training_samples']} listings")
    print(f"{'─'*40}")

    comps = result["comparable_listings"]
    if comps:
        print(f"\nComparable listings ({len(comps)} found):")
        for c in comps:
            dom = f"{c['days_on_market']}d" if c['days_on_market'] else "?"
            print(f"  ${c['price']:>8,.0f}  |  {c['year']} {c['title']}  |  {c['mileage']:,}km  |  {dom} on market")
            print(f"             {c['url']}")
    else:
        print("\nNo comparable listings in DB yet. Scrape more data.")


def cmd_status():
    from Session import get_session
    from sqlalchemy import text

    with get_session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM listings")).scalar()
        with_price = session.execute(text("SELECT COUNT(*) FROM listings WHERE price IS NOT NULL")).scalar()
        with_make = session.execute(text("SELECT COUNT(*) FROM listings WHERE make IS NOT NULL")).scalar()
        snapshots = session.execute(text("SELECT COUNT(*) FROM listing_snapshots")).scalar()
        makes = session.execute(text(
            "SELECT make, COUNT(*) as n FROM listings WHERE make IS NOT NULL "
            "GROUP BY make ORDER BY n DESC LIMIT 10"
        )).fetchall()

    print(f"\n=== DB Status ===")
    print(f"  Total listings:     {total:,}")
    print(f"  With price:         {with_price:,}")
    print(f"  With make/model:    {with_make:,}")
    print(f"  Snapshots:          {snapshots:,}")
    print(f"\nTop makes:")
    for make, count in makes:
        print(f"  {make:<20} {count:>5}")


def cmd_pipeline(max_pages: int = 5):
    cmd_scrape(max_pages=max_pages)
    cmd_train()


COMMANDS = {
    "scrape":      cmd_scrape,
    "bulk":        cmd_bulk,
    "sold_sweep":  cmd_sold_sweep,
    "train":       cmd_train,
    "value":       cmd_value,
    "pipeline":    cmd_pipeline,
    "status":      cmd_status,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    if cmd in ("scrape", "pipeline") and len(sys.argv) > 2:
        COMMANDS[cmd](max_pages=int(sys.argv[2]))
    else:
        COMMANDS[cmd]()