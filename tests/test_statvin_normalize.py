"""
tests/test_statvin_normalize.py

Pure unit tests for scrapers/statvin.py's normalize(). Fixtures below are
built from the field values actually observed live on stat.vin during
development (see the header comment in scrapers/statvin.py for the
inspection notes) — e.g. the "well-formed lot" fixture mirrors the real
2016 Lexus IS lot (VIN JTHBA1D27G5038442, lot 52347416, Copart, Front End
damage) that was inspected via WebFetch.

These tests exercise normalize() directly against hand-built raw card
dicts (the shape _parse_card() produces) rather than against live HTML,
since scrape_page()/run() could not be live-tested end-to-end from this
sandbox (Cloudflare blocked direct httpx access — see scrapers/statvin.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scrapers.statvin import normalize, _parse_slug, VIN_RE  # noqa: E402


def make_raw(**overrides) -> dict:
    """A well-formed raw card dict, based on the real 2016 Lexus IS lot
    observed live (lot 52347416, VIN JTHBA1D27G5038442, Copart, AL)."""
    base = {
        "url": "https://stat.vin/cars/jthba1d27g5038442-lexus-is-2016-200t-silver-inline-4",
        "slug": "jthba1d27g5038442-lexus-is-2016-200t-silver-inline-4",
        "vin": "JTHBA1D27G5038442",
        "lot_number": "52347416",
        "year": 2016,
        "make": "lexus",
        "model": "is",
        "trim": "200t-silver-inline-4",
        "damage_primary": "Front End",
        "damage_secondary": None,
        "run_drive_status": "Enhanced Vehicle",
        "auction_house": "Copart",
        "current_bid": 0,
        "seller_reserve": 3100.0,
        "auction_date_raw": "Thu 13 Aug 2026, 17:00",
        "odometer": None,
        "location": "AL - MONTGOMERY",
        "title_type": "AL - CERT OF TITLE-SALVAGE TITLE (P)",
        "image_urls": [
            "https://cdn13.stat.vin/2026/05/27/xxx/yyy/2/52347416/photo/photo_1.webp",
            "https://cdn13.stat.vin/2026/05/27/xxx/yyy/2/52347416/photo/photo_2.webp",
        ],
        "card_text": (
            "2016 LEXUS IS Lot: 52347416 VIN: JTHBA1D27G5038442 "
            "Front End Enhanced Vehicle Copart AL - MONTGOMERY "
            "AL - CERT OF TITLE-SALVAGE TITLE (P) Seller Reserve: $3,100 "
            "Thu 13 Aug 2026, 17:00"
        ),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Well-formed lot
# ---------------------------------------------------------------------------

def test_well_formed_lot_normalizes_fully():
    result = normalize(make_raw())
    assert result is not None

    assert result["source"] == "statvin"
    assert result["source_id"] == "52347416"
    assert result["listing_type"] == "salvage"
    assert result["vin"] == "JTHBA1D27G5038442"
    assert result["url"] == "https://stat.vin/cars/jthba1d27g5038442-lexus-is-2016-200t-silver-inline-4"

    assert result["year"] == 2016
    assert result["make"] == "lexus"
    assert result["model"] == "is"
    assert result["title"] == "2016 lexus is"

    # current_bid is 0 (no bids yet) -> falls back to seller_reserve
    assert result["price"] == 3100.0
    assert result["raw_json"]["price_source"] == "seller_reserve"

    assert result["damage_primary"] == "Front End"
    assert result["title_type"] == "AL - CERT OF TITLE-SALVAGE TITLE (P)"
    assert result["auction_house"] == "Copart"
    assert result["lot_number"] == "52347416"
    assert result["run_drive_status"] == "Enhanced Vehicle"
    assert result["auction_date"] == "Thu 13 Aug 2026, 17:00"
    assert result["auction_sale_status"] == "pending"
    assert result["location"] == "AL - MONTGOMERY"

    assert len(result["image_urls"]) == 2

    assert result["fingerprint"] is not None
    assert isinstance(result["fingerprint"], str)
    assert len(result["fingerprint"]) == 32  # md5 hexdigest


# ---------------------------------------------------------------------------
# 2. Lot missing photos
# ---------------------------------------------------------------------------

def test_lot_missing_photos_still_normalizes():
    raw = make_raw(image_urls=[])
    result = normalize(raw)

    assert result is not None
    assert result["image_urls"] == []
    # everything else should still be populated normally
    assert result["vin"] == "JTHBA1D27G5038442"
    assert result["price"] == 3100.0


# ---------------------------------------------------------------------------
# 3. Lot missing VIN
# ---------------------------------------------------------------------------

def test_lot_missing_vin_falls_back_to_lot_number_identity():
    raw = make_raw(vin=None)
    result = normalize(raw)

    assert result is not None
    assert result["vin"] is None
    # source_id falls back to lot_number since VIN is unavailable
    assert result["source_id"] == "52347416"
    # fingerprint is independent of VIN (year/make/model/mileage/location)
    assert result["fingerprint"] is not None
    assert result["make"] == "lexus"


def test_lot_missing_both_vin_and_lot_number_is_rejected():
    raw = make_raw(vin=None, lot_number=None)
    result = normalize(raw)

    # No stable identity to upsert on -> reject
    assert result is None


# ---------------------------------------------------------------------------
# 4. Lot with no numeric price
# ---------------------------------------------------------------------------

def test_lot_with_no_numeric_price_is_kept_with_null_price():
    raw = make_raw(current_bid=0, seller_reserve=None)
    result = normalize(raw)

    assert result is not None
    assert result["price"] is None
    assert result["raw_json"]["price_source"] is None
    # row is still usable — identity + damage/auction metadata intact
    assert result["vin"] == "JTHBA1D27G5038442"
    assert result["damage_primary"] == "Front End"
    assert result["auction_house"] == "Copart"


def test_current_bid_preferred_over_seller_reserve_when_both_present():
    raw = make_raw(current_bid=4200.0, seller_reserve=3100.0)
    result = normalize(raw)

    assert result["price"] == 4200.0
    assert result["raw_json"]["price_source"] == "current_bid"


def test_absurdly_large_parsed_price_is_treated_as_corrupted():
    # e.g. an accidentally-captured phone number or similar junk match
    raw = make_raw(current_bid=99_999_999.0, seller_reserve=None)
    result = normalize(raw)

    assert result["price"] is None


# ---------------------------------------------------------------------------
# Slug parsing (grounds year/make/model extraction — verified against 3
# real observed slugs, see scrapers/statvin.py docstring)
# ---------------------------------------------------------------------------

def test_parse_slug_matches_real_observed_examples():
    assert _parse_slug("jthba1d27g5038442-lexus-is-2016-200t-silver-inline-4") == {
        "vin": "JTHBA1D27G5038442",
        "make": "lexus",
        "model": "is",
        "year": 2016,
        "trim": "200t-silver-inline-4",
    }
    assert _parse_slug("58abk1gg9hu042798-lexus-es350-2017-es-silver-v6") == {
        "vin": "58ABK1GG9HU042798",
        "make": "lexus",
        "model": "es350",
        "year": 2017,
        "trim": "es-silver-v6",
    }
    assert _parse_slug("jthbf5c21c5173694-lexus-is-2012-gray-25l-6") == {
        "vin": "JTHBF5C21C5173694",
        "make": "lexus",
        "model": "is",
        "year": 2012,
        "trim": "gray-25l-6",
    }


def test_vin_re_matches_real_vin():
    assert VIN_RE.search("JTHBA1D27G5038442")


# ---------------------------------------------------------------------------
# Empty/garbage input
# ---------------------------------------------------------------------------

def test_normalize_handles_empty_dict():
    assert normalize({}) is None
