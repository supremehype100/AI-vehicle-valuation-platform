"""
scrapers/kijiji.py

Scrapes Kijiji vehicle listings using the __NEXT_DATA__ Apollo cache.

Actual data path (verified against live site):
  __NEXT_DATA__
    → props.pageProps.__APOLLO_STATE__
      → keys matching "AutosListing:<id>"
        → fields: __typename, id, title, description, imageCount,
                  imageUrls, categoryId, url, activationDate,
                  sortingDate, adSource, location, price, flags,
                  posterInfo, attributes

The Apollo cache is a flat dict — every listing is a top-level key
prefixed with "AutosListing:". No nested edges/nodes to traverse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from typing import Generator
from Features import is_valid_price
import httpx
from bs4 import BeautifulSoup


def clean_description(text: str | None) -> str:
    """Collapses line breaks, breaks up bullets, and outputs clean prose for ML."""
    if not text:
        return ""
    # Convert raw carriage returns and newlines to spaces
    cleaned = text.replace("\r", " ").replace("\n", " ")
    # Strip structural artifacts/bullets that cut off words
    cleaned = re.sub(r'[-\u2013\u2014*_•]+', ' ', cleaned)
    # Collapse multiple spaces into a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def fetch_full_description(listing_url: str, client: httpx.Client) -> str | None:
    """Navigates to the listing page to grab the full description from its local Apollo State."""
    try:
        # Re-use the master HEADERS from your script
        resp = client.get(listing_url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        tag = soup.find("script", id="__NEXT_DATA__")
        if not tag:
            return None

        vip_data = json.loads(tag.string)
        apollo = (
            vip_data.get("props", {})
            .get("pageProps", {})
            .get("__APOLLO_STATE__", {})
        )

        # Scan the flat Apollo state dictionary for the listing's full text
        for key, value in apollo.items():
            if key.startswith("AutosListing:") and isinstance(value, dict):
                full_desc = value.get("description")
                if full_desc:
                    return full_desc

    except Exception as e:
        logger.error("Failed to fetch detailed description from VIP %s: %s", listing_url, e)
    return None
logger = logging.getLogger(__name__)

BASE_URL = "https://www.kijiji.ca"

# Category + location slug — adjust city slug + location ID as needed.
# Format: /b-{category}/{city-slug}/c{category-id}l{location-id}
# Calgary vehicles: c174l1700199
# Canada-wide:      c174l0
SEARCH_URL = BASE_URL + "/b-cars-trucks/calgary/page-{page}/c174l1700199?for-sale-by=ownr"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.kijiji.ca",
}

VIN_RE = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def _fetch_page(page: int, client: httpx.Client) -> dict:
    """Fetch one search results page and return the raw __NEXT_DATA__ dict."""
    url = SEARCH_URL.format(page=page)
    resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag:
        logger.warning("No __NEXT_DATA__ found on page %d", page)
        return {}

    return json.loads(tag.string)


def _extract_listings_from_apollo(next_data: dict) -> list[dict]:
    """
    Pull all AutosListing entries from the Apollo cache.

    Apollo stores every queried object as a flat key:
        "AutosListing:1736024593" → { __typename, id, title, ... }

    We just filter for keys that start with "AutosListing:".
    """
    apollo: dict = (
        next_data
        .get("props", {})
        .get("pageProps", {})
        .get("__APOLLO_STATE__", {})
    )

    listings = []
    for key, value in apollo.items():
        if key.startswith("AutosListing:") and isinstance(value, dict):
            listings.append(value)

    return listings


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def _parse_attributes(raw_attrs) -> dict[str, str]:
    """
    Two attribute shapes exist in the wild:

    Shape A (canonicalName/canonicalValues — confirmed from live data):
        attributes = {
            "__typename": "AutosListingAttributes",
            "all": [
                { "__typename": "ListingAttributeV2",
                  "canonicalName": "carmake",
                  "canonicalValues": ["bmw"] },
            ]
        }

    Shape B (machineKey/localeSpecificValues — older listings):
        attributes = [
            { "machineKey": "carmake",
              "localeSpecificValues": [{ "value": "BMW" }] },
        ]
    """
    out = {}

    # Shape A — dict with an "all" list
    if isinstance(raw_attrs, dict):
        for attr in raw_attrs.get("all") or []:
            if not isinstance(attr, dict):
                continue
            key = attr.get("canonicalName")
            vals = attr.get("canonicalValues") or []
            if key and vals:
                out[key] = vals[0]
        return out

    # Shape B — bare list
    if isinstance(raw_attrs, list):
        for attr in raw_attrs:
            if not isinstance(attr, dict):
                continue
            key = attr.get("machineKey") or attr.get("key")
            if not key:
                continue
            locale_vals = attr.get("localeSpecificValues") or attr.get("values")
            if locale_vals and isinstance(locale_vals, list):
                out[key] = locale_vals[0].get("value", "")
            elif "value" in attr:
                out[key] = str(attr["value"])
        return out

    return out


def _parse_price(price_block: dict | None) -> float | None:
    """
    price block shape:  { "amount": 12500, "type": "FIXED" }
    or sometimes:       { "amount": 0, "type": "PLEASE_CONTACT" }
    """
    if not price_block:
        return None
    amount = price_block.get("amount")
    price_type = price_block.get("type", "")
    if price_type in ("PLEASE_CONTACT", "FREE", "SWAP_TRADE"):
        return None
    try:
        return float(amount) / 100 if amount else None
    except (TypeError, ValueError):
        return None


def _parse_location(loc_block: dict | None) -> str | None:
    """location block: { name, city, province, ... }"""
    if not loc_block:
        return None
    return loc_block.get("name") or loc_block.get("city")


def _extract_vin(text: str | None) -> str | None:
    if not text:
        return None
    m = VIN_RE.search(text.upper())
    return m.group(0) if m else None


def _make_fingerprint(year, make, model, mileage, city) -> str | None:
    if not all([year, make, model]):
        return None
    mileage_bucket = round((mileage or 0) / 5000) * 5000
    key = f"{year}:{str(make).lower()}:{str(model).lower()}:{mileage_bucket}:{str(city or '').lower()}"
    return hashlib.md5(key.encode()).hexdigest()


def normalize(raw: dict) -> dict:
    """
    Convert a raw Apollo AutosListing dict into a clean flat dict
    ready to be passed to RawListing(**result).

    Returns a plain dict rather than importing RawListing here
    so this module stays dependency-light and easily testable.
    """
    attrs = _parse_attributes(raw.get("attributes", []))
    location_str = _parse_location(raw.get("location"))
    price = _parse_price(raw.get("price"))
    if not is_valid_price(price):
        return None
    description = raw.get("description") or ""

    # Core vehicle attributes — canonicalName keys confirmed from live data
    year_raw = attrs.get("caryear") or attrs.get("year")
    make_raw = attrs.get("carmake") or attrs.get("make")
    model_raw = attrs.get("carmodel") or attrs.get("model")
    mileage_raw = attrs.get("carmileageinkms") or attrs.get("mileage") or attrs.get("kms")
    trim_raw = attrs.get("cartrim") or attrs.get("trim")
    color_raw = attrs.get("carcolor") or attrs.get("color") or attrs.get("colour")
    transmission = attrs.get("cartransmission") or attrs.get("transmission")
    fuel_type = attrs.get("carfueltype") or attrs.get("fueltype")
    body_type = attrs.get("carbodytype")
    drivetrain = attrs.get("drivetrain")

    try:
        year = int(year_raw) if year_raw else None
    except ValueError:
        year = None

    try:
        # Mileage often comes as "85,000 km" — strip non-digits
        mileage = int(re.sub(r'\D', '', str(mileage_raw))) if mileage_raw else None
    except ValueError:
        mileage = None

    # VIN: try dedicated attribute first, then scan description
    vin = attrs.get("vin") or _extract_vin(description) or _extract_vin(raw.get("title"))

    # posterInfo: { "type": "PRIVATE" | "DEALER", "name": "...", "memberId": "..." }
    poster = raw.get("posterInfo") or {}
    seller_type_raw = poster.get("type", "")
    seller_type = "dealer" if "DEALER" in seller_type_raw.upper() else "private"
    seller_id = str(poster.get("memberId") or poster.get("userId") or "")

    # imageUrls is a list of URL strings directly on the listing
    image_urls = raw.get("imageUrls") or []

    listing_url = raw.get("url") or ""
    if listing_url and not listing_url.startswith("http"):
        listing_url = BASE_URL + listing_url

    fingerprint = _make_fingerprint(year, make_raw, model_raw, mileage, location_str)

    return {
        "source": "kijiji",
        "source_id": str(raw.get("id", "")),
        "url": listing_url,
        "vin": vin,
        "fingerprint": fingerprint,

        "title": raw.get("title"),
        "make": make_raw,
        "model": model_raw,
        "year": year,
        "mileage": mileage,
        "trim": trim_raw,
        "color": color_raw,

        "price": price,
        "description": description or None,
        "status": "active",
        "seller_type": seller_type,
        "seller_id": seller_id or None,
        "location": location_str,

        "image_urls": image_urls,

        # Stash extra attrs in raw_json for later ML feature extraction
        "raw_json": {
            "transmission": transmission,
            "fuel_type": fuel_type,
            "body_type": body_type,
            "drivetrain": drivetrain,
            "flags": raw.get("flags"),
            "category_id": raw.get("categoryId"),
            "ad_source": raw.get("adSource"),
            "activation_date": raw.get("activationDate"),
            "sorting_date": raw.get("sortingDate"),
            "image_count": raw.get("imageCount"),
            "all_attributes": attrs,
            "_raw": raw,
        },

        "posted_at": raw.get("activationDate"),  # ISO string; parsed by ingest layer
    }


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape_page(page: int, client: httpx.Client) -> list[dict]:
    """Scrape one page, return list of normalised listing dicts."""
    try:
        next_data = _fetch_page(page, client)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %d on page %d", e.response.status_code, page)
        return []
    except Exception as e:
        logger.error("Failed to fetch page %d: %s", page, e)
        return []

    raw_listings = _extract_listings_from_apollo(next_data)

    if not raw_listings:
        logger.info("No AutosListing entries on page %d — stopping", page)
        return []

    logger.info("Page %d: found %d listings", page, len(raw_listings))
    return [normalize(r) for r in raw_listings if normalize(r) is not None]


def run(
        max_pages: int = 5,
        known_ids: set[str] | None = None,
) -> Generator[dict, None, None]:
    """Scrape up to max_pages pages of Kijiji vehicle listings with deep descriptions."""
    known_ids = known_ids or set()

    with httpx.Client(timeout=20) as client:
        for page in range(1, max_pages + 1):
            listings = scrape_page(page, client)

            if not listings:
                break

            listings.sort(
                key=lambda x: x.get("raw_json", {}).get("sorting_date") or "",
                reverse=True,
            )

            new_on_page = 0
            for listing in listings:
                if listing is None:
                    continue

                # Check if this is a brand new ad we need to process
                if listing["source_id"] not in known_ids:
                    new_on_page += 1
                    desc = listing.get("description") or ""

                    # If it's cut off, go deep-fetch it
                    if desc.endswith("..."):
                        logger.info("Truncated description found for ID %s. Fetching full text...",
                                    listing["source_id"])
                        full_raw_desc = fetch_full_description(listing["url"], client)

                        if full_raw_desc:
                            listing["description"] = clean_description(full_raw_desc)
                            listing["raw_json"]["_raw"]["description"] = full_raw_desc

                        # Anti-bot politeness throttling
                        time.sleep(random.uniform(1.0, 2.0))
                    else:
                        # Even if it's short and not cut off, clean it of line-break anomalies anyway
                        listing["description"] = clean_description(desc)

                yield listing

            if new_on_page == 0 and known_ids:
                logger.info("Page %d fully known — stopping early", page)
                break

            time.sleep(random.uniform(1.0, 2.5))