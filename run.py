"""
run.py — Main entry point.

Commands:
    python run.py scrape [pages]       — scrape Kijiji and AutoTrader
    python run.py scrape_fast [pages]  — quick scrape (default 1 page)
    python run.py bulk [pages]         — scrape all pages from all sources
    python run.py scrape_facebook      — scrape Facebook Marketplace feeds
    python run.py scrape_statvin [n]   — scrape stat.vin salvage listings
    python run.py sold_sweep           — mark removed listings
    python run.py train                — train the price model
    python run.py value                — interactive valuation prompt
    python run.py pipeline [pages]     — scrape + train in one shot
    python run.py status               — show DB stats
    (see COMMANDS below for the backfill, training and comparison commands)
"""

from __future__ import annotations
import os
import random
import sys
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PERSONAL_CSV = "vehicles_cleaned.csv"

FB_TARGET_URLS = [
    "https://www.facebook.com/marketplace/calgary/search?topLevelVehicleType=car_truck&query=Vehicles&category_id=546583916084032&exact=false&referral_ui_component=category_menu_item",  # Cars/Trucks
    "https://www.facebook.com/marketplace/?ref=app_tab",
    "https://www.facebook.com/marketplace/calgary/search?availability=in%20stock&query=cars&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=out%20of%20stock&query=cars&exact=false",
    "https://www.facebook.com/marketplace/calgary/search/?query=toyota",
    "https://www.facebook.com/marketplace/calgary/search?availability=out%20of%20stock&query=toyota&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=in%20stock&query=honda&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=out%20of%20stock&query=honda&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=in%20stock&query=chevrolet&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=out%20of%20stock&query=chevrolet&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=in%20stock&query=suv&exact=false",
    "https://www.facebook.com/marketplace/calgary/search?availability=out%20of%20stock&query=suv&exact=false",
]


def _print_counts(label: str, counts: dict):
    print(f"{label:<11}— New: {counts['new']} | Updated: {counts['updated']} | Errors: {counts['errors']}")


def cmd_scrape(max_pages: int = 5, bulk: bool = False):
    """Scrape Kijiji and AutoTrader. With bulk=True, ignore already-known ids
    so every page is fully processed."""
    from Session import create_tables, get_session
    from Ingest import upsert_batch, get_known_ids
    from kijiji import run as kijiji_run
    from Autotrader import run as autotrader_run

    create_tables()

    for label, source, scrape in (
        ("Kijiji", "kijiji", kijiji_run),
        ("AutoTrader", "autotrader", autotrader_run),
    ):
        with get_session() as session:
            known_ids = set() if bulk else get_known_ids(source, session)
        listings = list(scrape(max_pages=max_pages, known_ids=known_ids))
        logger.info("Scraped %d listings from %s", len(listings), label)
        with get_session() as session:
            counts = upsert_batch(listings, session)
        _print_counts(label, counts)


def cmd_scrape_fast(max_pages: int = 1):
    """Quick scrape — Kijiji + AutoTrader only, page 1 by default."""
    cmd_scrape(max_pages=max_pages)


def cmd_bulk(max_pages: int = 100):
    cmd_scrape(max_pages=max_pages, bulk=True)


def cmd_scrape_facebook(max_listings: int = 7):
    """Scrape every Facebook Marketplace feed in FB_TARGET_URLS, one at a time
    (they share a single headed Chrome profile)."""
    from Session import create_tables, get_session
    from Ingest import upsert_batch, get_known_ids
    from Facebook import run as facebook_run

    create_tables()

    with get_session() as session:
        known_ids = get_known_ids("facebook", session)

    for index, url in enumerate(FB_TARGET_URLS, 1):
        logger.info("Scraping Facebook feed %d/%d", index, len(FB_TARGET_URLS))
        try:
            listings = list(facebook_run(known_ids=known_ids, max_listings=max_listings, target_url=url))
            with get_session() as session:
                counts = upsert_batch(listings, session)
        except Exception:
            logger.exception("Facebook feed %d failed", index)
            continue
        known_ids.update(l["source_id"] for l in listings)
        _print_counts(f"Facebook {index}", counts)


def cmd_sold_sweep():
    from Session import get_session
    from Ingest import sold_sweep
    from kijiji import run as kijiji_run
    from Autotrader import run as autotrader_run

    logger.info("Running sold sweep...")

    kijiji_seen = {l["source_id"] for l in kijiji_run(max_pages=100)}
    at_seen = {l["source_id"] for l in autotrader_run(max_pages=100)}

    with get_session() as session:
        kc = sold_sweep("kijiji", kijiji_seen, session)
        ac = sold_sweep("autotrader", at_seen, session)

    print("\nSold sweep complete:")
    print(f"  Kijiji removed:     {kc['removed']}")
    print(f"  AutoTrader removed: {ac['removed']}")


def cmd_train():
    from Session import get_session
    from Model import train

    cv_path = PERSONAL_CSV if os.path.exists(PERSONAL_CSV) else None
    if cv_path:
        print("Using personal CSV as validation set")

    with get_session() as session:
        try:
            meta = train(session, cv_csv_path=cv_path)
        except ValueError as e:
            print(f"\n{e}")
            return

    print("\nModel trained:")
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
    try:
        year = int(input("Year (e.g. 2019): ").strip())
        mileage = int(input("Mileage in km (e.g. 80000): ").strip().replace(",", ""))
    except ValueError:
        print("\nYear and mileage must be whole numbers.")
        return
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
        sources = session.execute(text(
            "SELECT source, COUNT(*) as n FROM listings "
            "GROUP BY source ORDER BY n DESC"
        )).fetchall()
        makes = session.execute(text(
            "SELECT make, COUNT(*) as n FROM listings WHERE make IS NOT NULL "
            "GROUP BY make ORDER BY n DESC LIMIT 10"
        )).fetchall()

    print("\n=== DB Status ===")
    print(f"  Total listings:     {total:,}")
    print(f"  With price:         {with_price:,}")
    print(f"  With make/model:    {with_make:,}")
    print(f"  Snapshots:          {snapshots:,}")
    print("\nBy source:")
    for source, count in sources:
        print(f"  {source:<20} {count:>6}")
    print("\nTop makes:")
    for make, count in makes:
        print(f"  {make:<20} {count:>5}")


def cmd_backfill_descriptions():
    """Finds existing Kijiji listings with truncated descriptions and updates them."""
    from Session import get_session
    from kijiji import fetch_full_description, clean_description
    from sqlalchemy import text
    import httpx

    logger.info("Initializing historical Kijiji description backfill sweep...")

    # 1. Pull only active Kijiji rows where the description cuts off with an ellipsis
    with get_session() as session:
        rows = session.execute(text("""
                                    SELECT source_id, url
                                    FROM listings
                                    WHERE source = 'kijiji'
                                      AND status = 'active'
                                      AND description LIKE '%...'
                                    """)).fetchall()

    if not rows:
        logger.info("No truncated Kijiji listings found. Database is completely up to date!")
        return

    logger.info(f"Found {len(rows)} legacy listings requiring full description extraction.")

    updated_count = 0

    with httpx.Client(timeout=20) as client:
        for source_id, url in rows:
            if not url:
                continue

            logger.info(f"Deep fetching full text for listing ID: {source_id}...")

            # Use your new helper from your Kijiji file to request the VIP page
            full_raw_desc = fetch_full_description(url, client)

            if full_raw_desc:
                # Sanitize the line breaks and bullets
                cleaned_desc = clean_description(full_raw_desc)

                # 2. Open a precise transaction block to overwrite this specific item
                with get_session() as session:
                    session.execute(
                        text("""
                             UPDATE listings
                             SET description = :desc
                             WHERE source_id = :sid
                               AND source = 'kijiji'
                             """),
                        {"desc": cleaned_desc, "sid": source_id}
                    )

                updated_count += 1
                logger.info(f"Successfully backfilled ID {source_id}.")
            else:
                logger.warning(f"Could not retrieve full description for ID {source_id} (listing may be gone).")

            # Be nice to Kijiji's firewalls during the backfill process
            time.sleep(random.uniform(1.5, 3.0))

    print(f"\nSweep complete! Successfully upgraded {updated_count}/{len(rows)} descriptions.")


def cmd_pipeline(max_pages: int = 5):
    cmd_scrape(max_pages=max_pages)
    cmd_train()


def cmd_scrape_statvin(max_pages: int = 5):
    """Scrapes stat.vin salvage-auction listings. Rows are tagged
    listing_type='salvage' at ingest time and never mix into the private
    dataset. See scrapers/statvin.py's module docstring for known caveats
    (Cloudflare may block direct httpx access from some networks)."""
    from Session import create_tables, get_session
    from Ingest import upsert_batch, get_known_ids
    from scrapers.statvin import run as statvin_run

    create_tables()

    with get_session() as session:
        known_statvin = get_known_ids("statvin", session)

    statvin_listings = list(statvin_run(max_pages=max_pages, known_ids=known_statvin))
    logger.info("Scraped %d listings from stat.vin", len(statvin_listings))
    with get_session() as session:
        sc = upsert_batch(statvin_listings, session)
    print(f"\nstat.vin — New: {sc['new']} | Updated: {sc['updated']} | Errors: {sc['errors']}")


def cmd_backfill_conditions():
    """Runs the AI condition-extraction backfill with default settings
    (--dataset all --batch-size 50). For finer control (--dataset private,
    --limit N, etc.) run `python -m backfill.backfill_conditions` directly."""
    from backfill.backfill_conditions import main as backfill_main
    backfill_main([])


def _print_train_result(label: str, meta: dict):
    print(f"\n{label} trained:")
    for key in ("train_core_samples", "training_samples", "val_samples", "test_samples", "total_samples"):
        if key in meta:
            print(f"  {key}: {meta[key]}")
    for key in ("mae", "rmse", "r2", "median_abs_error", "median_pct_error", "mape"):
        if key in meta:
            print(f"  {key}: {meta[key]}")


def cmd_train_private_csv():
    """Model 1 — private listings, existing personal-CSV validation baseline (unmodified)."""
    from Session import get_session
    from training.private_csv import run as train_private_csv

    cv_path = PERSONAL_CSV if os.path.exists(PERSONAL_CSV) else None
    with get_session() as session:
        result = train_private_csv(session, cv_csv_path=cv_path)
    _print_train_result("Model 1 (private_csv)", result["meta"])


def cmd_train_private_traintest():
    """Model 2 — private listings, new time-aware train/test split."""
    from Session import get_session
    from training.private_traintest import run as train_private_traintest

    with get_session() as session:
        result = train_private_traintest(session)
    _print_train_result("Model 2 (private_traintest)", result["meta"])


def cmd_train_dealer_traintest():
    """Model 3 — dealer listings, new time-aware train/test split."""
    from Session import get_session
    from training.dealer_traintest import run as train_dealer_traintest

    with get_session() as session:
        result = train_dealer_traintest(session)
    _print_train_result("Model 3 (dealer_traintest)", result["meta"])


def cmd_train_salvage_traintest():
    """Model 4 — salvage listings, new time-aware train/test split."""
    from Session import get_session
    from training.salvage_traintest import run as train_salvage_traintest

    with get_session() as session:
        result = train_salvage_traintest(session)
    _print_train_result("Model 4 (salvage_traintest)", result["meta"])


def cmd_train_all():
    """Trains all four models that currently have enough data, skipping (and
    reporting) any that raise a friendly ValueError for insufficient data —
    e.g. salvage_traintest before the stat.vin scraper has run."""
    jobs = [
        ("Model 1 (private_csv)", cmd_train_private_csv),
        ("Model 2 (private_traintest)", cmd_train_private_traintest),
        ("Model 3 (dealer_traintest)", cmd_train_dealer_traintest),
        ("Model 4 (salvage_traintest)", cmd_train_salvage_traintest),
    ]
    for label, fn in jobs:
        try:
            fn()
        except ValueError as e:
            print(f"\nSkipped {label}: {e}")


def cmd_compare_models():
    """Generates the model comparison report (reports/model_comparison.md)
    from whatever's currently in the model_artifacts registry."""
    from Session import get_session
    from training import registry
    from training.evaluate import generate_comparison_report

    with get_session() as session:
        rows = []
        for model_key in ("private_csv", "private_traintest", "dealer_traintest", "salvage_traintest"):
            artifact = registry.get_latest(session, model_key)
            if artifact is None:
                continue
            rows.append({"model_name": f"{model_key} ({artifact.version})", **artifact.metrics})

    if not rows:
        print("No trained models found in the registry yet. Run `python run.py train_all` first.")
        return
    generate_comparison_report(rows)


def cmd_compare_private_fair():
    """Fair, apples-to-apples comparison: scores Model A (private_csv) and
    Model B (private_traintest) on the SAME held-out rows (Model B's own
    time-aware test split), each with its own encoders — unlike
    compare_models, which just tabulates each model's own self-reported
    metrics from two different evaluation sets. See
    training.evaluate.compare_private_baseline_vs_traintest for the documented
    caveat this doesn't (and can't, without retraining Model A) fully solve."""
    from Session import get_session
    from training.evaluate import compare_private_baseline_vs_traintest

    with get_session() as session:
        try:
            compare_private_baseline_vs_traintest(session)
        except ValueError as e:
            print(f"\n{e}")


COMMANDS = {
    "scrape":      cmd_scrape,
    "bulk":        cmd_bulk,
    "sold_sweep":  cmd_sold_sweep,
    "train":       cmd_train,
    "value":       cmd_value,
    "pipeline":    cmd_pipeline,
    "status":      cmd_status,
    "scrape_fast": cmd_scrape_fast,
    "scrape_facebook": cmd_scrape_facebook,
    "scrape_statvin": cmd_scrape_statvin,
    "backfill_kijiji": cmd_backfill_descriptions,
    "backfill_conditions": cmd_backfill_conditions,
    "train_private_csv": cmd_train_private_csv,
    "train_private_traintest": cmd_train_private_traintest,
    "train_dealer_traintest": cmd_train_dealer_traintest,
    "train_salvage_traintest": cmd_train_salvage_traintest,
    "train_all": cmd_train_all,
    "compare_models": cmd_compare_models,
    "compare_private_fair": cmd_compare_private_fair,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    if cmd in ("scrape", "scrape_fast", "pipeline", "bulk", "scrape_statvin") and len(sys.argv) > 2:
        COMMANDS[cmd](max_pages=int(sys.argv[2]))
    else:
        COMMANDS[cmd]()