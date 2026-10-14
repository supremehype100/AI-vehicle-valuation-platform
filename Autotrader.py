"""
autotrader.py

Scrapes AutoTrader.ca private seller listings for Calgary.

Key URL params:
  custtype=P        — private sellers only
  damaged_listing=include — include damaged/salvage listings
  size=100          — 100 results per page
  page={page}       — pagination
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Generator
from Features import is_valid_price
from kijiji import clean_description

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.autotrader.ca"
SEARCH_URL = (
    BASE_URL
    + "/cars/reg_ab/cit_calgary?offer=N%2CU&custtype=P&cy=CA"
    + "&damaged_listing=include&desc=1&sort=age&ustate=N%2CU"
    + "&zip=Calgary%2CAB&lat=51.05011&lon=-114.08529&atype=C"
    + "&inMarket=advancedSearch&size=100&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.autotrader.ca",
}

VIN_RE = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def _fetch_page(page: int, client: httpx.Client) -> dict:
    url = SEARCH_URL.format(page=page)
    resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag:
        logger.warning("No __NEXT_DATA__ on page %d", page)
        return {}

    return json.loads(tag.string)


def _extract_listings(next_data: dict) -> list[dict]:
    """
    AutoTrader CA puts listings directly on pageProps.listings (confirmed from live data).
    """
    page_props = next_data.get("props", {}).get("pageProps", {})

    # Confirmed path: pageProps.listings
    listings = page_props.get("listings", [])
    if listings:
        return listings

    # Fallback paths
    listings = (
        page_props
        .get("initialData", {})
        .get("searchResults", {})
        .get("listings", [])
    )
    if listings:
        return listings

    listings = page_props.get("initialData", {}).get("listings", [])
    if listings:
        return listings

    return _find_listings_recursive(next_data)


def _find_listings_recursive(data, depth=0) -> list[dict]:
    if depth > 6:
        return []
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "make" in data[0]:
            return data
        for item in data:
            result = _find_listings_recursive(item, depth + 1)
            if result:
                return result
    elif isinstance(data, dict):
        for v in data.values():
            result = _find_listings_recursive(v, depth + 1)
            if result:
                return result
    return []


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def _extract_vin(text: str | None) -> str | None:
    if not text:
        return None
    m = VIN_RE.search(str(text).upper())
    return m.group(0) if m else None


def normalize(raw: dict) -> dict | None:
    source_id = str(raw.get("id") or raw.get("identifier") or "")
    if not source_id:
        return None

    vehicle = raw.get("vehicle") or {}

    make = vehicle.get("make")
    model = vehicle.get("model")
    year = vehicle.get("modelYear")
    trim = vehicle.get("modelVersionInput") or vehicle.get("variant")
    transmission = vehicle.get("transmission")
    fuel_type = vehicle.get("fuel")

    # Mileage: "52,000 km" — strip non-digits
    mileage_raw = vehicle.get("mileageInKm") or ""
    try:
        mileage = int(re.sub(r'\D', '', str(mileage_raw))) if mileage_raw else None
    except ValueError:
        mileage = None

    # Price: { "priceFormatted": "$ 40,000" } — strip non-digits
    price_block = raw.get("price") or {}
    price_str = price_block.get("priceFormatted") or ""
    try:
        price = float(re.sub(r'[^\d.]', '', price_str)) if price_str else None
        price = price if price and price > 0 else None
    except ValueError:
        price = None

    if not is_valid_price(price):
        return None

    description = raw.get("description") or ""
    vin = _extract_vin(description)

    # Location: { "city": "Calgary", ... }
    loc = raw.get("location") or {}
    location = loc.get("city") or loc.get("region") or "Calgary"

    # Seller
    seller = raw.get("seller") or {}
    seller_id = str(seller.get("id") or seller.get("sellerId") or "")

    # Images: list of { "src": "...", ... } or strings
    image_urls = []
    for img in (raw.get("images") or []):
        if isinstance(img, str):
            image_urls.append(img)
        elif isinstance(img, dict):
            url = img.get("src") or img.get("url") or img.get("uri")
            if url:
                image_urls.append(url)

    listing_url = raw.get("url") or ""
    if listing_url and not listing_url.startswith("http"):
        listing_url = BASE_URL + listing_url

    # specialConditions may indicate salvage/damaged
    special = raw.get("specialConditions") or []
    damaged = any("damage" in str(s).lower() or "salvage" in str(s).lower() for s in special)

    return {
        "source": "autotrader",
        "source_id": source_id,
        "url": listing_url,
        "vin": vin,

        "title": f"{year} {make} {model}".strip(),
        "make": make,
        "model": model,
        "year": year,
        "mileage": mileage,
        "trim": trim,
        "color": None,  # not in search results, available on detail page

        "price": price,
        "description": description or None,
        "status": "active",
        "seller_type": "private",
        "seller_id": seller_id or None,
        "location": location,

        "image_urls": image_urls,

        "raw_json": {
            "body_type": vehicle.get("articleType"),
            "drivetrain": None,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "damaged": damaged,
            "engine_cc": vehicle.get("engineDisplacementInCCM"),
            "_raw": raw,
        },

        "posted_at": raw.get("activationDate") or raw.get("postedDate"),
    }


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape_page(page: int, client: httpx.Client) -> list[dict]:
    try:
        next_data = _fetch_page(page, client)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %d on page %d", e.response.status_code, page)
        return []
    except Exception as e:
        logger.error("Failed on page %d: %s", page, e)
        return []

    raw_listings = _extract_listings(next_data)
    if not raw_listings:
        logger.info("No listings on page %d — stopping", page)
        return []

    results = [n for r in raw_listings if (n := normalize(r)) is not None]
    logger.info("Page %d: found %d listings", page, len(results))
    return results


def run(
        max_pages: int = 5,
        known_ids: set[str] | None = None,
) -> Generator[dict, None, None]:
    known_ids = known_ids or set()

    with httpx.Client(timeout=20) as client:
        for page in range(1, max_pages + 1):
            listings = scrape_page(page, client)

            if not listings:
                break

            listings.sort(
                key=lambda x: x.get("posted_at") or "",
                reverse=True,
            )

            new_on_page = 0
            for listing in listings:
                if listing["source_id"] not in known_ids:
                    new_on_page += 1

                # Don't fetch detail pages; just flatten line breaks/bullets
                # in the description we already have.
                listing["description"] = clean_description(listing.get("description"))

                yield listing

            if new_on_page == 0 and known_ids:
                logger.info("Page %d fully known — stopping early", page)
                break

            time.sleep(random.uniform(2.0, 4.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Testing AutoTrader scraper...\n")
    results = list(run(max_pages=1))
    print(f"\nFound {len(results)} listings")

    if results:
        sample = results[0]
        print("\nFirst listing:")
        print(json.dumps({
            "source_id": sample["source_id"],
            "title": sample["title"],
            "price": sample["price"],
            "year": sample["year"],
            "make": sample["make"],
            "model": sample["model"],
            "mileage": sample["mileage"],
            "seller_type": sample["seller_type"],
            "location": sample["location"],
            "url": sample["url"],
        }, indent=2, default=str))

        nulls = [k for k in ["make", "model", "year", "price", "mileage"] if sample.get(k) is None]
        if nulls:
            print(f"\nNull fields: {nulls}")
            print("Top-level raw keys:", list(sample["raw_json"].get("_raw", {}).keys()))