"""
scrapers/statvin.py

Scrapes stat.vin — a free, no-login salvage/insurance-auction aggregator that
mirrors Copart and IAAI lot listings (VIN, damage, auction house, bid price,
title status, photos, etc). Rows produced here are tagged
listing_type="salvage" and feed the salvage_traintest pipeline.

=============================================================================
ACTUAL FINDINGS FROM LIVE INSPECTION (2026-08-12) — READ BEFORE EDITING
=============================================================================

Unlike kijiji.py / Autotrader.py, stat.vin is NOT a Next.js app and has NO
`__NEXT_DATA__` / `__NUXT__` / Apollo-style JSON cache embedded in the page.
This was confirmed by direct inspection of the homepage, a search-results
page, and a lot-detail page — the only embedded JSON on any of them is
schema.org JSON-LD (an `Organization` block on every page; `Vehicle` /
`HowTo` / `FAQPage` blocks additionally on `/cars/<slug>` detail pages).
There is therefore no single structured payload to lift listings out of;
this module extracts fields from rendered text instead (see "PARSING
STRATEGY" below).

--- Cloudflare: direct httpx from this sandbox was blocked -----------------
A plain `httpx.Client` request (with the same browser-like HEADERS used
successfully by kijiji.py/Autotrader.py) to both `https://stat.vin/` and
`https://stat.vin/robots.txt` returned HTTP 403 with a Cloudflare
"Attention Required" JS-challenge page from this sandbox's network egress.
A tool-mediated fetch (Claude's own WebFetch, which uses different fetch
infrastructure) *did* get through and is how the structural details below
were actually observed. This means:
  * `normalize()` below is unit-tested and known-good (see
    tests/test_statvin_normalize.py).
  * `_fetch_page()` / the raw HTML parsing regexes are written against real
    observed content (see below) but were NEVER exercised end-to-end
    against a live `httpx.Client.get()` response from this sandbox, because
    that request path is the one Cloudflare blocked. `scrape_page()`/`run()`
    are therefore UNVERIFIED end-to-end.
  * If the target deployment environment also gets Cloudflare-challenged,
    `_fetch_page()` is the (and the only) function that needs a different
    transport — e.g. curl_cffi's TLS-impersonation client, `cloudscraper`,
    or a real Chrome-channel Playwright session/profile like Facebook.py
    uses. Neither curl_cffi nor cloudscraper is in requirements.txt yet,
    and this sandbox doesn't have Playwright's browser binaries installed
    either (`playwright install` was not run here), so that fallback is
    also undemonstrated. All parsing logic (`_extract_cards_from_html`,
    `_parse_slug`, `normalize`, ...) is transport-agnostic and works on an
    HTML string, so swapping the transport later does not require touching
    parsing code.

--- robots.txt ---------------------------------------------------------
Fetched and quoted verbatim via WebFetch. Relevant bits:
  User-agent: *
  Allow: /
  Disallow: */ajax/ /public/ /livewire/ ... /api-ai/ /vin/ /embed/ (+more)
  Sitemap: https://stat.vin/sitemap.xml (+ 9 more)

  User-agent: GPTBot / ClaudeBot / PerplexityBot / OAI-SearchBot /
              Meta-ExternalAgent / Amazonbot
  Allow: /vin/
  Allow: /api-ai/
  Allow: /cars/
This scraper identifies with an ordinary browser User-Agent (matching the
rest of this codebase's convention), so it is governed by the `User-agent:
*` block — under which `/vin/` and `/api-ai/` (which look like they might
hold nicer structured data, going by the paths named) are Disallow'd for
us specifically, while `/auction-this-week` (search) and `/cars/` (lot
detail) are not in the generic Disallow list and are therefore allowed.
We deliberately stick to those two paths rather than spoof an AI-crawler
UA string to unlock `/vin/` or `/api-ai/`.

--- Search results: /auction-this-week --------------------------------
URL:      https://stat.vin/auction-this-week?vehicle_type=AUTOMOBILE&page={page}
Confirmed live: unfiltered (no `make=`) returns ~20 lots/page spanning many
makes (Mazda, Mercedes-Benz, Dodge, Ford, BMW, Toyota, ... observed on one
page), "51971 Vehicles found" total, and a page=2 link for pagination.
Each card links to a `/cars/<slug>` detail page and (per rendered-text
inspection) shows: lot number, VIN, year/make/model, primary (and
sometimes secondary) damage, a run/drive status label ("Run & Drive",
"Enhanced Vehicle", "Stationary", "Starts"), auction house (Copart/IAAI),
current bid and/or seller-reserve price, auction date, odometer, location,
a title/document-type string, and a thumbnail image.

--- Lot detail: /cars/<slug> -------------------------------------------
Slug format confirmed against 3 real examples:
  jthba1d27g5038442-lexus-is-2016-200t-silver-inline-4
  58abk1gg9hu042798-lexus-es350-2017-es-silver-v6
  jthbf5c21c5173694-lexus-is-2012-gray-25l-6
i.e. `<vin-lowercase>-<make>-<model>-<year>-<trim/color/engine tokens...>`.
This is a more reliable signal than any card text heuristic (see
`_parse_slug`), so year/make/model/vin are primarily derived from the URL
slug, not from scraped card text.
Detail pages additionally exposed (WebFetch inspection of one lot): keys
present (yes/no), body style, engine, transmission, drivetrain, fuel type,
color, a photo count with sequential CDN URLs
(".../<lot_number>/photo/photo_N.webp"), and a "Sales History" table of
prior auction attempts for the same lot (useful for `auction_sale_status`
in a future enhancement — NOT currently consumed; see below).

--- What this module actually fetches per run() -------------------------
To stay low-volume/respectful (stat.vin is a free public resource with no
login, robots.txt lists ten sitemaps implying a large dataset, and its own
robots.txt goes out of its way to allow named AI crawlers — i.e. this is a
site that wants to be legible to bots, but we should still not hammer it),
`run()` only fetches search-results pages (one request per page, 20 lots
each) and does NOT deep-fetch every `/cars/<slug>` detail page — the same
"zero-risk" choice Autotrader.py made for descriptions. `fetch_lot_detail()`
is provided standalone for future/optional enrichment but is not called
automatically.

=============================================================================
PARSING STRATEGY
=============================================================================
Because no raw HTML/CSS markup could be verified (WebFetch renders to text
for its summarizer and doesn't expose literal tags/classes; direct httpx
was Cloudflare-blocked), this module deliberately avoids depending on any
specific CSS class name. Instead:
  1. It anchors on `<a href="...cars/...">` tags — the one piece of real
     DOM structure that was directly confirmed (three literal hrefs were
     observed verbatim in the live search-page fetch).
  2. It derives year/make/model/vin from the URL slug (verified against 3
     real examples above), not from card text.
  3. It regex-extracts the remaining fields (lot number, damage, run/drive
     status, auction house, price, date, odometer, location, title type)
     from the plain text of the anchor's nearest ancestor that contains a
     VIN match — i.e. "whatever text block wraps this specific lot",
     without needing to know its tag/class name.
This is inherently a best-effort heuristic layer and is the most likely
part of this file to need adjustment once someone can inspect real page
markup (e.g. via browser devtools, or once a Cloudflare-capable transport
is wired up in `_fetch_page`). Every extractor degrades to `None` on
non-match rather than raising, and `normalize()` is written to tolerate
missing fields (see PRICE POLICY below).

=============================================================================
PRICE POLICY (do not reuse Features.is_valid_price for this source)
=============================================================================
`Features.is_valid_price` rejects anything under $1,000 or over $200,000
and a small placeholder-price set. That is correct for private/dealer
listings but wrong here: a pre-auction lot legitimately shows a *current
bid of $0* (no bids placed yet — not "worth zero"), and a genuinely
burned-out or flood-totaled shell can legitimately sell for a few hundred
dollars. Rejecting either case would silently discard real salvage data.

Policy implemented in `normalize()`:
  * Prefer the current bid when it is a positive number (most faithful
    "current market signal" for an active auction).
  * If the bid is 0/absent, fall back to the seller-reserve figure (if
    positive) so a nonzero pricing signal isn't discarded outright.
  * If neither is available, `price` is left as `None` — the row is still
    kept (VIN/damage/auction metadata alone is useful), matching the task
    note that "the site might not expose a numeric price for some lots."
  * `raw_json["price_source"]` records which of the two (or neither) was
    used, so downstream consumers can filter reserve-only rows out if they
    want higher-fidelity price data.
  * A generous upper sanity cap (500,000) is applied only to catch
    obviously corrupted parses (e.g. an accidentally-captured phone
    number), not to enforce a private-market price range.
  * `Features.load_training_data()` already gates on `is_valid_price` at
    *training* time for the private-listing pipeline, so keeping
    unpriced/low-priced salvage rows here does not leak them into that
    model; a salvage-specific training pipeline can apply its own filter.

A row is only rejected (normalize() returns None) when it has no derivable
stable identity at all — i.e. both lot_number and VIN are missing, so there
is nothing to upsert on.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from typing import Generator

import httpx
from bs4 import BeautifulSoup

from config import LISTING_TYPE_SALVAGE

logger = logging.getLogger(__name__)

BASE_URL = "https://stat.vin"

# Unfiltered (no `make=`) so one crawl covers every make; confirmed live to
# return ~20 lots/page across many makes. `vehicle_type=AUTOMOBILE` excludes
# motorcycles/boats which the site also auctions but which this codebase's
# schema (make/model/trim/mileage) isn't shaped for.
SEARCH_URL = BASE_URL + "/auction-this-week?vehicle_type=AUTOMOBILE&page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://stat.vin/",
}

VIN_RE = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')

# ---------------------------------------------------------------------------
# Best-effort text-extraction vocabularies (see "PARSING STRATEGY" above —
# these are Copart/IAAI's publicly-known damage/run-drive vocabulary plus
# the specific labels actually observed live; not guaranteed exhaustive).
# ---------------------------------------------------------------------------
DAMAGE_TYPES = [
    "Minor Dent/Scratches", "Minor Dents/Scratches", "Biohazard/Chemical",
    "Storm/Flood", "Water/Flood", "Fuel/Mechanical", "Frame Damage",
    "Undercarriage", "All Over", "Left Side", "Right Side", "Rear End",
    "Front End", "Rollover", "Vandalism", "Stripped", "Top/Roof",
    "Electrical", "Mechanical", "Normal Wear", "Not Actual", "Side",
    "Flood", "Rear", "Burn", "Hail",
]

RUN_DRIVE_LABELS = [
    "Run & Drive", "Run and Drive", "Enhanced Vehicle", "Stationary",
    "Does Not Run", "Not Actual", "Starts",
]

_BID_RE = re.compile(
    r'(?:Current\s*Bid|Bid\s*Amount|Bid)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)',
    re.IGNORECASE,
)
_RESERVE_RE = re.compile(
    r'(?:Seller\s*Reserve|Reserve\s*Price|Reserve)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)',
    re.IGNORECASE,
)
_LOT_RE = re.compile(r'\bLot\b\s*(?:#|No\.?|Number)?\s*:?\s*(\d{6,9})', re.IGNORECASE)
_ODO_RE = re.compile(r'([\d,]+)\s*mi\b', re.IGNORECASE)
_AUCTION_HOUSE_RE = re.compile(r'\b(Copart|IAAI)\b', re.IGNORECASE)
# "Thu 13 Aug 2026, 17:00" style — parsed downstream by dateutil (Ingest.py).
_AUCTION_DATE_RE = re.compile(
    r'\b((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4},?\s*\d{1,2}:\d{2})',
)
# "XX - Some Words" — used for both location ("AL - MONTGOMERY") and title
# type ("AL - CERT OF TITLE-SALVAGE TITLE (P)"); classified by TITLE keyword.
_STATE_DASH_RE = re.compile(
    r'\b([A-Z]{2})\s*-\s*([A-Z][A-Za-z0-9 /\-\(\)\.]{2,60}?)(?=\s{2,}|\s+[A-Z]{2}\s*-|\s+\d{1,2}:\d{2}|$)'
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _fetch_page(page: int, client: httpx.Client) -> str:
    """Fetch one search-results page and return the raw HTML text.

    NOTE: exercised only via WebFetch during development (see module
    docstring) — a plain httpx.Client request to stat.vin from this sandbox
    was Cloudflare-challenged (HTTP 403). Left as httpx to match the rest of
    this codebase's convention; swap the transport here (not the parsing
    functions below) if that also happens in the real deployment.
    """
    url = SEARCH_URL.format(page=page)
    resp = client.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
    resp.raise_for_status()
    return resp.text


def fetch_lot_detail(url: str, client: httpx.Client) -> str | None:
    """Fetch a single /cars/<slug> detail page. Not called by run() by
    default (see "What this module actually fetches" in the module
    docstring) — provided for optional future enrichment (photos, engine,
    transmission, sales history, etc.)."""
    try:
        resp = client.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception as e:
        logger.error("Failed to fetch lot detail %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Parse — slug (most reliable signal, see module docstring)
# ---------------------------------------------------------------------------

def _parse_slug(slug: str) -> dict:
    """
    Parse `<vin>-<make>-<model>-<year>-<trim/color/engine tokens...>`.
    Verified against 3 real observed slugs — see module docstring.
    """
    if not slug:
        return {}
    parts = slug.split("-")
    if not parts:
        return {}

    vin_candidate = parts[0].upper()
    vin = vin_candidate if VIN_RE.fullmatch(vin_candidate) else None

    year_idx = None
    for i, p in enumerate(parts[1:], start=1):
        if re.fullmatch(r'(19|20)\d{2}', p):
            year_idx = i
            break

    out: dict = {"vin": vin}
    if year_idx is None or year_idx < 2:
        return out

    out["make"] = parts[1]
    out["model"] = "-".join(parts[2:year_idx]) if year_idx > 2 else None
    out["year"] = int(parts[year_idx])
    trim = "-".join(parts[year_idx + 1:])
    out["trim"] = trim or None
    return out


# ---------------------------------------------------------------------------
# Parse — search results page -> list of raw card dicts
# ---------------------------------------------------------------------------

def _extract_vin(text: str | None) -> str | None:
    if not text:
        return None
    m = VIN_RE.search(text.upper())
    return m.group(0) if m else None


def _search_amount(regex: re.Pattern, text: str) -> float | None:
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_damage(text: str) -> tuple[str | None, str | None]:
    lower = text.lower()
    found: list[str] = []
    for label in sorted(DAMAGE_TYPES, key=len, reverse=True):
        if label.lower() in lower and label not in found:
            found.append(label)
        if len(found) >= 2:
            break
    primary = found[0] if found else None
    secondary = found[1] if len(found) > 1 else None
    return primary, secondary


def _extract_run_drive(text: str) -> str | None:
    lower = text.lower()
    for label in RUN_DRIVE_LABELS:
        if label.lower() in lower:
            return label
    return None


def _extract_auction_house(text: str) -> str | None:
    m = _AUCTION_HOUSE_RE.search(text)
    return m.group(1).title() if m else None


def _extract_lot_number(text: str) -> str | None:
    m = _LOT_RE.search(text)
    return m.group(1) if m else None


def _extract_odometer(text: str) -> int | None:
    m = _ODO_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_auction_date(text: str) -> str | None:
    m = _AUCTION_DATE_RE.search(text)
    return m.group(1) if m else None


def _extract_location_and_title(text: str) -> tuple[str | None, str | None]:
    location = None
    title_type = None
    for m in _STATE_DASH_RE.finditer(text):
        chunk = f"{m.group(1)} - {m.group(2)}".strip()
        if "title" in chunk.lower():
            if title_type is None:
                title_type = chunk
        else:
            if location is None:
                location = chunk
        if location and title_type:
            break
    return location, title_type


def _parse_card(a_tag, href: str) -> dict:
    """Build a raw card dict from a single `<a href=".../cars/...">` tag,
    walking up to the nearest ancestor whose text contains a VIN (see
    "PARSING STRATEGY" in the module docstring)."""
    node = a_tag
    text = node.get_text(" ", strip=True)
    depth = 0
    while not VIN_RE.search(text.upper()) and depth < 5:
        parent = node.find_parent()
        if parent is None:
            break
        node = parent
        text = node.get_text(" ", strip=True)
        depth += 1

    slug_match = re.search(r'/cars/([^/?#]+)', href)
    slug = slug_match.group(1) if slug_match else ""
    slug_fields = _parse_slug(slug)

    images: list[str] = []
    for img in node.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if src and src not in images:
            images.append(src)

    location, title_type = _extract_location_and_title(text)
    damage_primary, damage_secondary = _extract_damage(text)
    current_bid = _search_amount(_BID_RE, text)
    seller_reserve = _search_amount(_RESERVE_RE, text)

    return {
        "url": href,
        "slug": slug,
        "vin": slug_fields.get("vin") or _extract_vin(text),
        "lot_number": _extract_lot_number(text),
        "year": slug_fields.get("year"),
        "make": slug_fields.get("make"),
        "model": slug_fields.get("model"),
        "trim": slug_fields.get("trim"),
        "damage_primary": damage_primary,
        "damage_secondary": damage_secondary,
        "run_drive_status": _extract_run_drive(text),
        "auction_house": _extract_auction_house(text),
        "current_bid": current_bid,
        "seller_reserve": seller_reserve,
        "auction_date_raw": _extract_auction_date(text),
        "odometer": _extract_odometer(text),
        "location": location,
        "title_type": title_type,
        "image_urls": images,
        "card_text": text,
    }


def _extract_cards_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    seen_hrefs: set[str] = set()
    cards: list[dict] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/cars/" not in href:
            continue
        if href.startswith("/"):
            href = BASE_URL + href
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        try:
            cards.append(_parse_card(a, href))
        except Exception as e:
            logger.error("Failed to parse card for %s: %s", href, e)

    return cards


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def _to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(x) -> int | None:
    f = _to_float(x)
    return int(f) if f is not None else None


def _make_fingerprint(year, make, model, mileage, location) -> str | None:
    if not all([year, make, model]):
        return None
    bucket = round((mileage or 0) / 5000) * 5000
    key = f"{year}:{str(make).lower()}:{str(model).lower()}:{bucket}:{str(location or '').lower()}"
    return hashlib.md5(key.encode()).hexdigest()


_MAX_SANE_PRICE = 500_000  # generous cap — see PRICE POLICY in module docstring


def _resolve_price(current_bid: float | None, seller_reserve: float | None) -> tuple[float | None, str | None]:
    if current_bid is not None and current_bid > 0 and current_bid <= _MAX_SANE_PRICE:
        return current_bid, "current_bid"
    if seller_reserve is not None and seller_reserve > 0 and seller_reserve <= _MAX_SANE_PRICE:
        return seller_reserve, "seller_reserve"
    return None, None


def _compose_description(raw: dict) -> str | None:
    parts = [
        raw.get("damage_primary") and f"{raw['damage_primary']} damage.",
        raw.get("damage_secondary") and f"Secondary: {raw['damage_secondary']}.",
        raw.get("run_drive_status") and f"{raw['run_drive_status']}.",
        raw.get("title_type") and f"Title: {raw['title_type']}.",
    ]
    text = " ".join(p for p in parts if p)
    return text or None


def normalize(raw: dict) -> dict | None:
    """
    Convert a raw card dict (see `_parse_card`) into the flat listing dict
    shape shared with kijiji.normalize / Autotrader.normalize. Returns None
    only when there is no derivable stable identity (no lot_number and no
    VIN) — see "PRICE POLICY" above for why a missing/zero price does NOT
    cause rejection.
    """
    vin = raw.get("vin")
    if vin:
        vin = vin.upper()
        if not VIN_RE.fullmatch(vin):
            vin = None

    lot_number = raw.get("lot_number")
    source_id = str(lot_number or vin or "")
    if not source_id:
        return None

    year = _to_int(raw.get("year"))
    make = raw.get("make")
    model = raw.get("model")
    mileage = _to_int(raw.get("odometer"))
    location = raw.get("location")

    current_bid = _to_float(raw.get("current_bid"))
    seller_reserve = _to_float(raw.get("seller_reserve"))
    price, price_source = _resolve_price(current_bid, seller_reserve)

    url = raw.get("url") or ""
    if url and not url.startswith("http"):
        url = BASE_URL + url

    fingerprint = _make_fingerprint(year, make, model, mileage, location)

    title_bits = [str(year) if year else None, make, model]
    title = " ".join(b for b in title_bits if b) or raw.get("slug") or None

    return {
        "source": "statvin",
        "source_id": source_id,
        "url": url or None,
        "vin": vin,
        "fingerprint": fingerprint,
        "listing_type": LISTING_TYPE_SALVAGE,

        "title": title,
        "make": make,
        "model": model,
        "year": year,
        "mileage": mileage,
        "trim": raw.get("trim"),
        "color": raw.get("color"),

        "price": price,
        "description": _compose_description(raw),
        "status": "active",
        "seller_type": None,
        "seller_id": None,
        "location": location,

        "image_urls": raw.get("image_urls") or [],

        "raw_json": {
            "lot_number": lot_number,
            "slug": raw.get("slug"),
            "current_bid": current_bid,
            "seller_reserve": seller_reserve,
            "price_source": price_source,
            "run_drive_status": raw.get("run_drive_status"),
            "auction_house_raw": raw.get("auction_house"),
            "card_text": raw.get("card_text"),
            "_raw": raw,
        },

        "posted_at": None,

        # --- salvage-specific ---
        "damage_primary": raw.get("damage_primary"),
        "damage_secondary": raw.get("damage_secondary"),
        "title_type": raw.get("title_type"),
        "auction_house": raw.get("auction_house"),
        "auction_date": raw.get("auction_date_raw"),
        "lot_number": lot_number,
        "run_drive_status": raw.get("run_drive_status"),
        # Rows come from the live "auction-this-week" feed (upcoming/active
        # auctions), so price is a current bid/reserve, not a hammer price,
        # unless a future enhancement deep-fetches sales-history off the
        # detail page (see module docstring).
        "auction_sale_status": "pending",
    }


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape_page(page: int, client: httpx.Client) -> list[dict]:
    try:
        html = _fetch_page(page, client)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP %d on page %d", e.response.status_code, page)
        return []
    except Exception as e:
        logger.error("Failed to fetch page %d: %s", page, e)
        return []

    raw_cards = _extract_cards_from_html(html)
    if not raw_cards:
        logger.info("No lot cards found on page %d — stopping", page)
        return []

    results = [n for c in raw_cards if (n := normalize(c)) is not None]
    logger.info("Page %d: found %d lots", page, len(results))
    return results


def run(
        max_pages: int = 5,
        known_ids: set[str] | None = None,
) -> Generator[dict, None, None]:
    """Scrape up to max_pages pages of stat.vin salvage-auction search
    results. Conservative delays (2-4s) — this is a free, no-login public
    resource and we want to stay well within respectful crawl volume."""
    known_ids = known_ids or set()

    with httpx.Client(timeout=20) as client:
        for page in range(1, max_pages + 1):
            listings = scrape_page(page, client)

            if not listings:
                break

            new_on_page = 0
            for listing in listings:
                if listing["source_id"] not in known_ids:
                    new_on_page += 1
                yield listing

            if new_on_page == 0 and known_ids:
                logger.info("Page %d fully known — stopping early", page)
                break

            time.sleep(random.uniform(2.0, 4.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Testing stat.vin scraper...\n")
    print("NOTE: direct httpx access to stat.vin was Cloudflare-blocked from")
    print("the sandbox this scraper was developed in (see module docstring).")
    print("If this smoke test finds 0 listings, that's the likely cause —")
    print("check for an HTTP 403 in the log line above before assuming a")
    print("parsing bug.\n")

    results = list(run(max_pages=1))
    print(f"\nFound {len(results)} listings")

    if results:
        sample = results[0]
        import json
        print("\nFirst listing:")
        print(json.dumps({
            "source_id": sample["source_id"],
            "title": sample["title"],
            "vin": sample["vin"],
            "price": sample["price"],
            "year": sample["year"],
            "make": sample["make"],
            "model": sample["model"],
            "damage_primary": sample["damage_primary"],
            "auction_house": sample["auction_house"],
            "auction_date": sample["auction_date"],
            "title_type": sample["title_type"],
            "url": sample["url"],
        }, indent=2, default=str))

        nulls = [k for k in ["make", "model", "year", "price", "vin"] if sample.get(k) is None]
        if nulls:
            print(f"\nNull fields: {nulls}")
