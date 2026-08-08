import asyncio
import hashlib
import logging
import random
import sys
import re
from pathlib import Path
from Features import is_valid_price
from config import (
    FB_LISTING_DELAY_MIN_SECONDS,
    FB_LISTING_DELAY_MAX_SECONDS,
    FB_PAGE_LOAD_WAIT_MIN_MS,
    FB_PAGE_LOAD_WAIT_MAX_MS,
)
from classification.classify import DEALER_NAME_RE, DEALER_DESC_RE, dealer_match_reason

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

CHROME_USER_DATA = r"./fb_chrome_profile_fresh"
CHROME_PROFILE = "Default"
TARGET_URL = "https://www.facebook.com/marketplace/calgary/search?topLevelVehicleType=car_truck&query=Vehicles&category_id=546583916084032&exact=false&referral_ui_component=category_menu_item"

Path(CHROME_USER_DATA).mkdir(parents=True, exist_ok=True)

DISCOVERED_LISTING_IDS = set()

VIN_RE = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')

VALID_MAKES = {
    "Abarth", "Acura", "Alfa Romeo", "Alpine", "Aston Martin", "Audi", "Austin", "Austin-Healey", "AMC", "Autozam",
    "Bentley", "BMW", "BYD", "Bugatti", "Buick", "Cadillac", "Caterham", "Chevrolet", "Chrysler", "Citroen", "Cupra",
    "Dacia", "Daewoo", "Daihatsu", "Datsun", "De Tomaso", "Dodge", "Donkervoort", "DS", "Eagle", "Efini", "Eunos",
    "Ferrari", "Fiat", "Fisker", "Ford", "Genesis", "Geo", "GMC", "Hino", "Holden", "Honda", "Hummer", "Hyundai",
    "Infiniti", "Innocenti", "Isuzu", "Iveco", "Jaguar", "Jeep", "Kia", "Koenigsegg", "KTM", "Lada", "Lamborghini",
    "Lancia", "Land Rover", "Lexus", "Lincoln", "Lotus", "Lucid", "Mahindra", "Maserati", "Maybach", "Mazda",
    "McLaren", "Mercedes-Benz", "Mercury", "MG", "Mini", "Mitsubishi", "Mitsuoka", "Morgan", "Morris", "Nissan",
    "Noble", "NSU", "Oldsmobile", "Opel", "Pagani", "Peugeot", "Plymouth", "Polestar", "Pontiac", "Porsche",
    "Prince", "Proton", "Ram", "Renault", "Reliant", "Rimac", "Rivian", "Rolls-Royce", "Rover", "Saab", "Saleen",
    "Saturn", "Scion", "SEAT", "Skoda", "Smart", "Spyker", "SsangYong", "Subaru", "Suzuki", "Tata", "Tesla",
    "Tommykaira", "Toyota", "Toyopet", "Triumph", "TVR", "UAZ", "Vauxhall", "Venturi", "VinFast", "Vista",
    "Volkswagen", "Volvo", "Wartburg", "Wiesmann", "Zenvo"
}

SORTED_MAKES = sorted(list(VALID_MAKES), key=len, reverse=True)


def _check_dealer_match(seller_name: str | None, description: str | None) -> str | None:
    return dealer_match_reason(seller_name, description)


def _extract_year_make_model(title: str) -> tuple[int | None, str | None, str | None]:
    if not title:
        return None, None, None

    # 1. Hard-remove Facebook noise FIRST (Prefixes and Suffixes)
    title = re.sub(r'^\(\d+\)\s*', '', title)  # removes (2), (10), etc
    title = re.sub(r'^Marketplace\s*[\s–\-—|•]*', '', title, flags=re.IGNORECASE)  # removes "Marketplace – "
    title = re.sub(r'[\s–\-—|•]+\s*Facebook.*$', '', title, flags=re.IGNORECASE)  # removes " | Facebook"

    title = title.replace("|", " ")
    title = re.sub(r'\s+', " ", title).strip()

    # 2. Extract Year
    year = None
    parts = title.split()
    for part in parts:
        if part.isdigit() and len(part) == 4 and part.startswith(("19", "20")):
            year = int(part)
            break

    # 3. Detect Make
    detected_make = None
    title_lower = title.lower()

    for brand in SORTED_MAKES:
        brand_lower = brand.lower()
        brand_pattern = re.escape(brand_lower).replace(r'\-', r'[\s\-]?')
        pattern = r'(?:^|[\s().,/?!@#%^&*_\-—–])' + brand_pattern + r'(?:$|[\s().,/?!@#%^&*_\-—–])'

        if re.search(pattern, title_lower):
            detected_make = brand
            break

    if not detected_make:
        return year, None, None

    # 4. Extract model safely via token filtering
    model_parts = title.split()

    # Filter out year and make tokens cleanly
    model_parts = [
        p for p in model_parts
        if p.lower() != detected_make.lower() and p != str(year)
    ]

    # Join and clean up any lingering inner/outer punctuation symbols
    model = " ".join(model_parts).strip(" -|—–•.")

    # Trim to a clean 2-word model (e.g., "Atenza Sport Wagon" -> "Atenza Sport")
    final_parts = model.split()
    model = " ".join(final_parts[:2]) if final_parts else "Unknown"

    return year, detected_make, model.title()

def parse_raw_listing(extracted: dict) -> dict | None:
    title = extracted.get("title", "")
    description = extracted.get("description", "")
    seller_name = extracted.get("seller_name", "")
    listing_id = extracted.get("id", "Unknown")

    raw_price = extracted.get("price")
    price = None
    if raw_price is not None:
        try:
            if isinstance(raw_price, (int, float)):
                price = float(raw_price)
            else:
                cleaned_price_str = re.sub(r'[^\d.]', '', str(raw_price))
                price = float(cleaned_price_str) if cleaned_price_str else None
        except (ValueError, TypeError):
            price = None

    if not is_valid_price(price):
        logger.info(" Dropped listing ID %s: Invalid price value detected (Raw: '%s', Parsed: %s)", listing_id,
                    raw_price, price)
        return None

    if re.search(r'\bparts?\b', title, re.IGNORECASE):
        logger.info(" Dropped listing ID %s: Title contains parts/salvage keywords ('%s')", listing_id, title)
        return None

    # Dealer listings are no longer discarded here — they're kept and routed
    # to the dealer dataset (see classification/classify.py + Ingest.py)
    # instead of being thrown away. We still tag *why* for observability.
    dealer_reason = _check_dealer_match(seller_name, description)
    seller_type = "dealer" if dealer_reason else "private"
    if dealer_reason:
        logger.info(" Listing ID %s flagged as dealer (kept, routed to dealer dataset). Reason: %s",
                    listing_id, dealer_reason)

    year, make, model = _extract_year_make_model(title)

    if not year or not make:
        reasons = []
        if not year: reasons.append("Missing or unparseable production year")
        if not make: reasons.append("No supported automaker brand match detected")
        logger.info(" Dropped listing ID %s: Structural check failed (%s). Title was: '%s'", listing_id,
                    " & ".join(reasons), title)
        return None

    raw_mileage = extracted.get("mileage", "")
    mileage = None
    if raw_mileage and raw_mileage != "Not Listed":
        try:
            cleaned_mileage_str = re.sub(r'[^\d]', '', str(raw_mileage))
            if cleaned_mileage_str:
                mileage = int(cleaned_mileage_str)
        except Exception as e:
            logger.info(" Mileage conversion failed crashed on string %r: %s", raw_mileage, e)
            mileage = None

    return {
        "source": "facebook",
        "source_id": listing_id,
        "title": title,
        "price": price,
        "year": year,
        "make": make,
        "model": model,
        "mileage": mileage,
        "description": description,
        "seller_name": seller_name,
        "seller_type": seller_type,
        "raw_json": extracted,
        "status": extracted.get("status", "active")
    }


def _extract_vin(text: str | None) -> str | None:
    if not text:
        return None
    m = VIN_RE.search(str(text).upper())
    return m.group(0) if m else None


def _make_fingerprint(year, make, model, mileage, city) -> str | None:
    if not all([year, make, model]):
        return None
    bucket = round((mileage or 0) / 500) * 500
    key = f"{year}:{str(make).lower()}:{str(model).lower()}:{bucket}:{str(city or '').lower()}"
    return hashlib.md5(key.encode()).hexdigest()


def normalize_visible(extracted: dict, listing_id: str) -> dict | None:
    title = extracted.get("title", "")
    if re.search(r'\bparts?\b', title, re.IGNORECASE):
        logger.info("❌ normalize_visible skipped ID %s: Title contains parts keywords.", listing_id)
        return None

    description = extracted.get("description", "")
    seller_name = extracted.get("seller_name", "")

    # Dealer listings are kept (routed to the dealer dataset), not dropped.
    dealer_reason = _check_dealer_match(seller_name, description)
    seller_type = "dealer" if dealer_reason else "private"

    price_str = extracted.get("price", "")
    try:
        price = float(re.sub(r'[^\d.]', '', price_str)) if price_str else None
        price = price if price and price > 0 else None
    except ValueError:
        price = None

    if not is_valid_price(price):
        logger.info("❌ normalize_visible skipped ID %s: Price out of bounds (%s).", listing_id, price_str)
        return None

    mileage_str = extracted.get("mileage", "")
    try:
        mileage = int(re.sub(r'\D', '', mileage_str)) if mileage_str else None
    except ValueError:
        mileage = None

    year, make, model = _extract_year_make_model(title)
    if not year or not make:
        logger.info("❌ normalize_visible skipped ID %s: Missing year/make identifiers ('%s').", listing_id, title)
        return None

    return {
        "source": "facebook",
        "source_id": listing_id,
        "url": f"https://www.facebook.com/marketplace/item/{listing_id}",
        "vin": _extract_vin(description),
        "fingerprint": _make_fingerprint(year, make, model, mileage, extracted.get("location")),
        "title": title,
        "make": make,
        "model": model,
        "year": year,
        "mileage": mileage,
        "trim": None,
        "color": None,
        "price": price,
        "description": description or None,
        "status": extracted.get("status", "active"),
        "seller_type": seller_type,
        "seller_id": None,
        "location": extracted.get("location"),
        "image_urls": [],
        "raw_json": {
            "body_type": None,
            "drivetrain": None,
            "fuel_type": None,
            "transmission": None,
            "seller_name": seller_name,
        },
        "posted_at": None,
    }


async def extract_visible_details(page, listing_id):
    url = f"https://www.facebook.com/marketplace/item/{listing_id}/"
    print(f"\n🔎 Extracting visible elements for ID: {listing_id}...")

    extracted_data = {
        "title": "Unknown Vehicle",
        "price": "Contact for Price",
        "mileage": "Not Listed",
        "location": "Calgary (Generic)",
        "seller_name": "Private Seller",
        "description": "No description found.",
        "status": "active"
    }

    try:
        await page.goto(url, wait_until="domcontentloaded")
        # Page-load render wait — a correctness safeguard, not the inter-listing
        # pause, so it stays at its original value (see config.py).
        await page.wait_for_timeout(random.randint(FB_PAGE_LOAD_WAIT_MIN_MS, FB_PAGE_LOAD_WAIT_MAX_MS))

        try:
            buttons = await page.query_selector_all(
                "div[role='button']:has-text('See more'), span:has-text('See more'), div:has-text('See more')")
            for btn in reversed(buttons):
                box = await btn.bounding_box()
                if box and box['width'] < 300 and box['height'] < 60:
                    if await btn.is_visible():
                        await btn.scroll_into_view_if_needed()
                        await btn.click(force=True)
                        await page.wait_for_timeout(1500)
                        break
        except Exception:
            pass

        body_element = await page.query_selector("body")
        page_text = await body_element.inner_text() if body_element else ""
        current_url = page.url
        if "unavailable_product=1" in current_url:
            extracted_data["status"] = "removed"
            return extracted_data

        if re.search(r"this listing is no longer available|listing has expired", page_text, re.IGNORECASE):
            extracted_data["status"] = "removed"
            return extracted_data

        if re.search(r'\bsold\b|\bpending\b', page_text[:500], re.IGNORECASE):
            extracted_data["status"] = "sold"
        if not page_text:
            return extracted_data

        title_raw = await page.title()
        if title_raw:
            extracted_data["title"] = title_raw.strip()

        price_match = re.search(r'(?:CA\s*)?\$([\d,]+)\s*\n', page_text)
        if price_match:
            extracted_data["price"] = f"${price_match.group(1)}"
        else:
            price_fallback = re.search(r'\$?([\d,]+)\s*(?=\nListed)', page_text)
            if price_fallback:
                extracted_data["price"] = f"${price_fallback.group(1)}"

        loc_match = re.search(r'Listed\s+.*?\s+in\s+([^\n]+)', page_text)
        if loc_match:
            raw_location = loc_match.group(1).strip()
            extracted_data["location"] = raw_location.split("-")[-1].strip() if "-" in raw_location else raw_location

        seller_match = re.search(r'\n([^\n]+)\n\(\d+\)\nJoined Facebook', page_text)
        if seller_match:
            extracted_data["seller_name"] = seller_match.group(1).strip()
        else:
            seller_fallback = re.search(r'Seller details\n([^\n]+)', page_text)
            if seller_fallback:
                extracted_data["seller_name"] = seller_fallback.group(1).strip()

        desc_match = re.search(
            r"Seller's description\n(.*?\n?)(?=Location is approximate|Seller information|Seller details)", page_text,
            re.DOTALL)
        if desc_match:
            extracted_data["description"] = desc_match.group(1).strip()
        else:
            fallback = re.search(r'Details\n(.*?)(?=Location is approximate|Seller information|Seller details|\Z)',
                                 page_text, re.DOTALL)
            if fallback:
                extracted_data["description"] = fallback.group(1).strip()

        specs_match = re.search(r'Driven\s+([\d,]+\s*(?:km|miles|k))', page_text, re.IGNORECASE)
        strict_desc_match = re.search(r'\b(\d{1,3}[,\s]\d{3}|\d{4,6})\s*(?:km|kms|miles)\b',
                                      extracted_data["description"], re.IGNORECASE)
        k_desc_match = re.search(r'\b(\d{2,3}\s*k\b\s*(?:km)?)\b', extracted_data["description"], re.IGNORECASE)
        loose_raw_match = re.search(r'\b(\d{3}[,\s]\d{3})\s*km\b', page_text, re.IGNORECASE)

        if specs_match:
            extracted_data["mileage"] = specs_match.group(1).strip()
        elif strict_desc_match:
            extracted_data["mileage"] = f"{strict_desc_match.group(1).strip()} km"
        elif k_desc_match:
            extracted_data["mileage"] = k_desc_match.group(1).strip().lower().replace("k", ",000 km")
        elif loose_raw_match:
            extracted_data["mileage"] = f"{loose_raw_match.group(1).strip()} km"

        m_str = extracted_data["mileage"]
        if m_str.strip() == "65 km" or (m_str.startswith("65") and len(m_str) <= 6):
            extracted_data["mileage"] = "Not Listed"

    except Exception as e:
        print(f" Element parsing error on ID {listing_id}: {e}")
    return extracted_data


async def run_scraper(target_url: str | None = None):
    from playwright.async_api import async_playwright
    if target_url is None:
        target_url = TARGET_URL
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(Path(CHROME_USER_DATA).resolve()),
            headless=False,
            ignore_default_args=["--enable-automation"],
            # ⬇️ TWEAKED ARGS: Added focus prevention flags ⬇️
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-startup-window",  # Spawns window in the background
                "--inactive"            # Tells the OS not to bring this window to the foreground
            ],
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Edmonton"
        )

        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(target_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)

        links = await page.query_selector_all("a[href*='/marketplace/item/']")
        for link in links:
            href = await link.get_attribute("href")
            if href:
                match = re.search(r'/item/(\d+)', href)
                if match:
                    DISCOVERED_LISTING_IDS.add(match.group(1))

        target_queue = list(DISCOVERED_LISTING_IDS)[:4]
        for index, item_id in enumerate(target_queue, 1):
            await extract_visible_details(page, item_id)
            # Pause between checking listings — configurable, see config.py.
            await asyncio.sleep(random.uniform(FB_LISTING_DELAY_MIN_SECONDS, FB_LISTING_DELAY_MAX_SECONDS))

        await context.close()


def run(max_listings: int = 30, known_ids: set[str] | None = None, dealer_ids: set[str] | None = None,
        target_url: str | None = None):
    known_ids = known_ids or set()
    dealer_ids = dealer_ids or set()
    url_to_scrape = target_url or TARGET_URL

    results = asyncio.run(_run_and_collect(max_listings=max_listings, target_url=url_to_scrape))

    for listing in results:
        if listing and listing["source_id"] not in known_ids:
            if listing.get("seller_id") not in dealer_ids:
                yield listing


async def _run_and_collect(max_listings: int, target_url: str) -> list[dict]:
    from playwright.async_api import async_playwright
    collected = []

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(Path(CHROME_USER_DATA).resolve()),
            headless=False,
            ignore_default_args=["--enable-automation"],
            # ⬇️ TWEAKED ARGS: Added focus prevention flags ⬇️
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-startup-window",  # Spawns window in the background
                "--inactive"            # Tells the OS not to bring this window to the foreground
            ],
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Edmonton",
        )

        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(target_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)

        links = await page.query_selector_all("a[href*='/marketplace/item/']")
        listing_ids = set()
        for link in links:
            href = await link.get_attribute("href")
            if href:
                match = re.search(r'/item/(\d+)', href)
                if match:
                    listing_ids.add(match.group(1))

        logger.info("Found %d listing IDs on the page", len(listing_ids))

        for listing_id in list(listing_ids):
            if len(collected) >= max_listings:
                break

            extracted = await extract_visible_details(page, listing_id)
            if extracted:
                extracted["id"] = listing_id
                clean_listing = parse_raw_listing(extracted)
                if clean_listing:
                    print("=" * 70)
                    print(f" VEHICLE ACCEPTED: {clean_listing['title']}")
                    print(f" PRICE:              ${clean_listing['price']}")
                    print(f" YEAR:               {clean_listing['year']}")
                    print(f" MAKE:               {clean_listing['make']}")
                    print(f" MODEL:              {clean_listing['model']}")
                    print(f" MILEAGE:            {clean_listing['mileage']} km" if clean_listing[
                        'mileage'] else " MILEAGE:            Not Listed")
                    print(f" SELLER:             {clean_listing['seller_name']}")
                    print("=" * 70)

                    collected.append(clean_listing)

            # Pause between checking listings — configurable, see config.py.
            await asyncio.sleep(random.uniform(FB_LISTING_DELAY_MIN_SECONDS, FB_LISTING_DELAY_MAX_SECONDS))
        await context.close()

    return collected


if __name__ == "__main__":
    asyncio.run(run_scraper())