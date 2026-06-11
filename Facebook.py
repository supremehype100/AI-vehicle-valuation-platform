import asyncio
import hashlib
import logging
import random
import sys
from pathlib import Path
from Features import is_valid_price

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

# We will collect listing IDs straight from the visible page links
DISCOVERED_LISTING_IDS = set()

import re

VIN_RE = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')

DEALER_NAME_RE = re.compile(
    r'\b(auto|motors|cars|dealership|sales|group|ltd|inc|corp|'
    r'center|centre|imports|wholesale|leasing)\b',
    re.IGNORECASE
)

DEALER_DESC_RE = re.compile(
    r'financing available|warranty included|hst not included|'
    r'tax not included|certified pre.owned|cpo|contact dealer|'
    r'test drive|our lot|visit us',
    re.IGNORECASE
)


def _is_dealer(seller_name: str | None, description: str | None) -> bool:
    if seller_name and DEALER_NAME_RE.search(seller_name):
        return True
    if description and DEALER_DESC_RE.search(description):
        return True
    return False


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
    # Parse price — "$12,500" → 12500.0
    price_str = extracted.get("price", "")
    try:
        price = float(re.sub(r'[^\d.]', '', price_str)) if price_str else None
        price = price if price and price > 0 else None
    except ValueError:
        price = None
    # Parse mileage — "120,000 km" → 120000
    mileage_str = extracted.get("mileage", "")
    try:
        mileage = int(re.sub(r'\D', '', mileage_str)) if mileage_str else None
    except ValueError:
        mileage = None

    title = extracted.get("title", "")
    if re.search(r'\bparts?\b', title, re.IGNORECASE):
        return None
    description = extracted.get("description", "")
    seller_name = extracted.get("seller_name", "")

    # Skip dealers
    if _is_dealer(seller_name, description):
        return None
    if not is_valid_price(price):
        return None
    # Parse year/make/model from title — "2019 Toyota Camry SE"
    parts = title.strip().split()
    try:
        year = int(parts[0]) if parts and parts[0].isdigit() else None
    except (ValueError, IndexError):
        year = None
    make = parts[1] if len(parts) > 1 else None
    model = parts[2] if len(parts) > 2 else None
    if not year and not make:
        logger.debug("Skipping non-vehicle listing %s: %s", listing_id, title)
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
        "status": "active",
        "seller_type": "private",
        "seller_id": None,  # not available from visible scrape
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
    """
    Rips the full visible text block from the page and uses structural
    text landmarks to cleanly isolate price, neighborhood, and seller details.
    """
    url = f"https://www.facebook.com/marketplace/item/{listing_id}/"
    print(f"\n️ Extracting visible elements for ID: {listing_id}...")

    # -------------------------------------------------------------------------
    # CRITICAL FIX: Baseline State Instantiation (Prevents Local Variable Crashes)
    # -------------------------------------------------------------------------
    extracted_data = {
        "title": "Unknown Vehicle",
        "price": "Contact for Price",
        "mileage": "Not Listed",
        "location": "Calgary (Generic)",
        "seller_name": "Private Seller",
        "description": "No description found."
    }

    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(random.randint(5000, 7000))

        # Expand Text Blocks ("See more" Action Loop)
        try:
            buttons = await page.query_selector_all(
                "div[role='button']:has-text('See more'), span:has-text('See more'), div:has-text('See more')")
            for btn in reversed(buttons):
                box = await btn.bounding_box()
                if box and box['width'] < 300 and box['height'] < 60:
                    if await btn.is_visible():
                        await btn.scroll_into_view_if_needed()
                        await btn.click(force=True)
                        print(f"   Target hit! Clicked expansion button (Size: {box['width']}x{box['height']})")
                        await page.wait_for_timeout(1500)
                        break
        except Exception:
            pass

        # Capture complete page footprint text stream safely
        body_element = await page.query_selector("body")
        page_text = await body_element.inner_text() if body_element else ""

        if not page_text:
            print(" Blank page text returned.")
            return

        # -------------------------------------------------------------------------
        # PARSING LOOPS (Safely Updating Dict Fields Instead of Dynamic Variables)
        # -------------------------------------------------------------------------

        # 1. Title Extraction
        title_raw = await page.title()
        if title_raw and "Marketplace" in title_raw:
            title_clean = title_raw.split(" - ")[0] if " - " in title_raw else title_raw
            extracted_data["title"] = title_clean.replace("Facebook Marketplace", "").replace("Marketplace", "").strip(
                " –| ")

        # 2. Price Extraction
        price_match = re.search(r'(?:CA\s*)?\$([\d,]+)\s*\n', page_text)
        if price_match:
            extracted_data["price"] = f"${price_match.group(1)}"
        else:
            price_fallback = re.search(r'\$?([\d,]+)\s*(?=\nListed)', page_text)
            if price_fallback:
                extracted_data["price"] = f"${price_fallback.group(1)}"

        # 3. Neighborhood Extraction
        loc_match = re.search(r'Listed\s+.*?\s+in\s+([^\n]+)', page_text)
        if loc_match:
            raw_location = loc_match.group(1).strip()
            extracted_data["location"] = raw_location.split("-")[-1].strip() if "-" in raw_location else raw_location

        # 4. Seller Name Extraction
        seller_match = re.search(r'\n([^\n]+)\n\(\d+\)\nJoined Facebook', page_text)
        if seller_match:
            extracted_data["seller_name"] = seller_match.group(1).strip()
        else:
            seller_fallback = re.search(r'Seller details\n([^\n]+)', page_text)
            if seller_fallback:
                extracted_data["seller_name"] = seller_fallback.group(1).strip()

        # 5. Description Block Extraction
        desc_match = re.search(
            r"Seller's description\n(.*?\n?)(?=Location is approximate|Seller information|Seller details)",
            page_text, re.DOTALL)
        if desc_match:
            extracted_data["description"] = desc_match.group(1).strip()
        else:
            # No structured description — grab text between Details and Location
            fallback = re.search(
                r'Details\n(.*?)(?=Location is approximate|Seller information|Seller details|\Z)',
                page_text, re.DOTALL)
            if fallback:
                extracted_data["description"] = fallback.group(1).strip()
        # 6. Mileage Extraction Priority Stack
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

        # Sanity checking to filter the 65km radius marker
        m_str = extracted_data["mileage"]
        if m_str.strip() == "65 km" or (m_str.startswith("65") and len(m_str) <= 6):
            extracted_data["mileage"] = "Not Listed"

        # -------------------------------------------------------------------------
        # TERMINAL LOGGER OUTPUT VIEW
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f" VEHICLE:       {extracted_data['title']}")
        print(f" PRICE:         {extracted_data['price']}")
        print(f" MILEAGE:       {extracted_data['mileage']}")
        print(f"️ NEIGHBORHOOD:  {extracted_data['location']}")
        print(f" SELLER:        {extracted_data['seller_name']}")
        print(f" DESCRIPTION:\n{extracted_data['description']}...")
        print(f" DIRECT LINK:   {url}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f" Robust Landmark layer error on ID {listing_id}: {e}")
    return extracted_data
async def run_scraper():
    from playwright.async_api import async_playwright

    print("Initializing browser footprint (1080p Profile)...")
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(Path(CHROME_USER_DATA).resolve()),
            headless=False,
            ignore_default_args=["--enable-automation"],
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Edmonton"
        )

        page = context.pages[0] if context.pages else await context.new_page()

        logger.info("Loading Marketplace dashboard view...")
        await page.goto(TARGET_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)

        # Scrape Phase: Extract listing links directly from the visible grid layout anchor tags
        logger.info("Gathering vehicle target anchors from current viewport...")
        links = await page.query_selector_all("a[href*='/marketplace/item/']")
        for link in links:
            href = await link.get_attribute("href")
            if href:
                # Isolate the core numeric ID block out of the long URL path string
                import re
                match = re.search(r'/item/(\d+)', href)
                if match:
                    DISCOVERED_LISTING_IDS.add(match.group(1))

        logger.info(f"Successfully harvested {len(DISCOVERED_LISTING_IDS)} unique listings from view.")

        # Deep-Dive Phase: Process the first 4 items using our fresh visual text extractor
        target_queue = list(DISCOVERED_LISTING_IDS)[:4]
        print(f" Starting Deep Extraction Loop for {len(target_queue)} vehicles...")

        for index, item_id in enumerate(target_queue, 1):
            print(f"\n[Processing Item {index}/{len(target_queue)}]")
            await extract_visible_details(page, item_id)
            cooldown = random.uniform(6.0, 10.0)
            print(f"  Cooling down browser profile for {cooldown:.2f}s...")
            await asyncio.sleep(cooldown)

        await context.close()

def run(max_listings: int = 50, known_ids: set[str] | None = None, dealer_ids: set[str] | None = None):
    known_ids = known_ids or set()
    dealer_ids = dealer_ids or set()

    results = asyncio.run(_run_and_collect(max_listings))
    for listing in results:
        if listing and listing["source_id"] not in known_ids:
            if listing.get("seller_id") not in dealer_ids:
                yield listing


async def _run_and_collect(max_listings: int) -> list[dict]:
    from playwright.async_api import async_playwright

    collected = []

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(Path(CHROME_USER_DATA).resolve()),
            headless=False,  # keep False — FB detects headless easily
            ignore_default_args=["--enable-automation"],
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1920, "height": 1080},
            locale="en-CA",
            timezone_id="America/Edmonton",
        )

        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(TARGET_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)

        # Collect listing IDs
        links = await page.query_selector_all("a[href*='/marketplace/item/']")
        listing_ids = set()
        for link in links:
            href = await link.get_attribute("href")
            if href:
                match = re.search(r'/item/(\d+)', href)
                if match:
                    listing_ids.add(match.group(1))

        logger.info("Found %d listing IDs", len(listing_ids))

        for listing_id in list(listing_ids)[:max_listings]:
            extracted = await extract_visible_details(page, listing_id)
            if extracted:
                normalized = normalize_visible(extracted, listing_id)
                if normalized:
                    collected.append(normalized)
            await asyncio.sleep(random.uniform(6.0, 10.0))

        await context.close()

    return collected

if __name__ == "__main__":
    asyncio.run(run_scraper())