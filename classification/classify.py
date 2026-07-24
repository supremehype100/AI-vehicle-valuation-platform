"""
classification/classify.py

Single source of truth for deciding whether a scraped listing is
private / dealer / salvage (config.LISTING_TYPE_*). Every scraper and the
ingest layer should call `classify_listing()` rather than re-deriving this
from seller_type or source ad hoc — that's how private/dealer/salvage
contamination has historically crept in (e.g. Kijiji sets seller_type
correctly but nothing downstream ever filtered on it).

The dealer name/description regexes were originally written inside
Facebook.py; they're reusable across every source (a dealer selling on
Kijiji looks the same as a dealer selling on Facebook), so they live here and
Facebook.py imports them back instead of keeping a second copy.
"""
from __future__ import annotations
import re

from config import LISTING_TYPE_PRIVATE, LISTING_TYPE_DEALER, LISTING_TYPE_SALVAGE

DEALER_NAME_RE = re.compile(
    r'\b(auto|motors|cars|dealership|sales|group|ltd|inc|corp|'
    r'center|centre|imports|wholesale|leasing)\b',
    re.IGNORECASE,
)

DEALER_DESC_RE = re.compile(
    r'financing available|warranty included|hst not included|'
    r'tax not included|certified pre.owned|cpo|contact dealer|'
    r'test drive|our lot|visit us',
    re.IGNORECASE,
)

# Sources whose listings are salvage-auction data by construction, regardless
# of any seller_type value they happen to carry.
SALVAGE_SOURCES = {"statvin"}


def dealer_match_reason(seller_name: str | None, description: str | None) -> str | None:
    """Returns a human-readable reason if dealer signals are found, else None."""
    if seller_name and DEALER_NAME_RE.search(seller_name):
        return f"Dealer name pattern matched inside seller name '{seller_name}'"
    if description and DEALER_DESC_RE.search(description):
        return "Commercial sales vocabulary found inside description text"
    return None


def classify_listing(listing_dict: dict) -> str:
    """
    Decide the listing_type for a normalized listing dict (the same shape
    produced by kijiji.normalize / Autotrader.normalize / Facebook.normalize_visible
    / scrapers/statvin.normalize).

    Precedence:
      1. Source is a known salvage-auction source -> salvage
      2. Explicit auction fields present (damage_primary/lot_number) -> salvage
      3. seller_type already says "dealer" -> dealer
      4. Dealer name/description regex match -> dealer
      5. Otherwise -> private
    """
    source = (listing_dict.get("source") or "").lower()
    if source in SALVAGE_SOURCES:
        return LISTING_TYPE_SALVAGE
    if listing_dict.get("damage_primary") or listing_dict.get("lot_number"):
        return LISTING_TYPE_SALVAGE

    if (listing_dict.get("seller_type") or "").lower() == "dealer":
        return LISTING_TYPE_DEALER

    raw_json = listing_dict.get("raw_json") or {}
    seller_name = listing_dict.get("seller_name") or raw_json.get("seller_name")
    reason = dealer_match_reason(seller_name, listing_dict.get("description"))
    if reason:
        return LISTING_TYPE_DEALER

    return LISTING_TYPE_PRIVATE
