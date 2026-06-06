from __future__ import annotations
import logging
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from models import Listing, DBListingSnapshot as ListingSnapshot

logger = logging.getLogger(__name__)


def utcnow():
    return datetime.now(tz=timezone.utc)


def _has_changed(existing: Listing, incoming: dict) -> bool:
    return (
        existing.price != incoming.get("price")
        or existing.status != incoming.get("status", "active")
        or existing.mileage != incoming.get("mileage")
    )


def upsert(listing_dict: dict, session: Session) -> str:
    """
    Insert or update one listing.
    Returns 'new', 'updated', or 'unchanged'.
    """
    source = listing_dict["source"]
    source_id = listing_dict["source_id"]
    now = utcnow()

    existing = session.execute(
        select(Listing).where(
            Listing.source == source,
            Listing.source_id == source_id,
        )
    ).scalar_one_or_none()

    if existing is None:
        # Parse posted_at if it's a string
        posted_at = listing_dict.get("posted_at")
        if isinstance(posted_at, str):
            try:
                from dateutil import parser as dateparser
                posted_at = dateparser.parse(posted_at)
            except Exception:
                posted_at = None

        row = Listing(
            source=source,
            source_id=source_id,
            url=listing_dict.get("url"),
            vin=listing_dict.get("vin"),
            fingerprint=listing_dict.get("fingerprint") or Listing.make_fingerprint(
                listing_dict.get("year"),
                listing_dict.get("make"),
                listing_dict.get("model"),
                listing_dict.get("mileage"),
                listing_dict.get("location"),
            ),
            title=listing_dict.get("title"),
            make=listing_dict.get("make"),
            model=listing_dict.get("model"),
            year=listing_dict.get("year"),
            mileage=listing_dict.get("mileage"),
            trim=listing_dict.get("trim"),
            color=listing_dict.get("color"),
            price=listing_dict.get("price"),
            description=listing_dict.get("description"),
            status=listing_dict.get("status", "active"),
            seller_type=listing_dict.get("seller_type"),
            seller_id=listing_dict.get("seller_id"),
            location=listing_dict.get("location"),
            first_seen=now,
            last_seen=now,
            posted_at=posted_at,
            days_on_market=0,
            image_urls=listing_dict.get("image_urls") or [],
            raw_json=listing_dict.get("raw_json") or {},
        )
        session.add(row)
        session.flush()  # get row.id

        session.add(ListingSnapshot(
            listing_id=row.id,
            source=source,
            source_id=source_id,
            scraped_at=now,
            price=row.price,
            status=row.status,
            mileage=row.mileage,
            description=row.description,
        ))
        return "new"

    elif _has_changed(existing, listing_dict):
        existing.price = listing_dict.get("price")
        existing.status = listing_dict.get("status", "active")
        existing.mileage = listing_dict.get("mileage")
        existing.last_seen = now
        existing.days_on_market = (now - existing.first_seen.replace(tzinfo=timezone.utc)).days
        existing.image_urls = listing_dict.get("image_urls") or existing.image_urls
        existing.raw_json = listing_dict.get("raw_json") or existing.raw_json

        session.add(ListingSnapshot(
            listing_id=existing.id,
            source=source,
            source_id=source_id,
            scraped_at=now,
            price=existing.price,
            status=existing.status,
            mileage=existing.mileage,
            description=existing.description,
        ))
        return "updated"

    else:
        existing.last_seen = now
        existing.days_on_market = (now - existing.first_seen.replace(tzinfo=timezone.utc)).days
        return "unchanged"


def upsert_batch(listings: list[dict], session: Session) -> dict:
    counters = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
    for listing in listings:
        try:
            result = upsert(listing, session)
            counters[result] += 1
        except Exception as e:
            logger.error("Failed to upsert %s:%s — %s", listing.get("source"), listing.get("source_id"), e)
            session.rollback()
            counters["errors"] += 1
    session.commit()
    return counters


def get_known_ids(source: str, session: Session) -> set[str]:
    """Return all source_ids we already have for this source — used for early exit."""
    rows = session.execute(
        select(Listing.source_id).where(Listing.source == source)
    ).scalars().all()
    return set(rows)