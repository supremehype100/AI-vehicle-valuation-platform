"""Tests for classification/classify.py — private/dealer/salvage separation."""
from classification.classify import classify_listing, dealer_match_reason
from config import LISTING_TYPE_PRIVATE, LISTING_TYPE_DEALER, LISTING_TYPE_SALVAGE


def test_plain_private_listing():
    listing = {
        "source": "kijiji",
        "seller_type": "private",
        "description": "Great car, low km, one owner.",
    }
    assert classify_listing(listing) == LISTING_TYPE_PRIVATE


def test_seller_type_dealer_wins():
    listing = {"source": "kijiji", "seller_type": "dealer", "description": ""}
    assert classify_listing(listing) == LISTING_TYPE_DEALER


def test_dealer_name_pattern_in_seller_name():
    listing = {
        "source": "facebook",
        "seller_type": None,
        "seller_name": "Calgary Auto Sales Ltd",
        "description": "Nice truck",
    }
    assert classify_listing(listing) == LISTING_TYPE_DEALER


def test_dealer_vocabulary_in_description():
    listing = {
        "source": "facebook",
        "seller_type": None,
        "seller_name": "John Smith",
        "description": "Financing available, warranty included, visit us today!",
    }
    assert classify_listing(listing) == LISTING_TYPE_DEALER


def test_seller_name_nested_in_raw_json_still_detected():
    listing = {
        "source": "facebook",
        "seller_type": None,
        "raw_json": {"seller_name": "Downtown Motors Inc"},
        "description": "",
    }
    assert classify_listing(listing) == LISTING_TYPE_DEALER


def test_statvin_source_is_always_salvage_even_with_private_seller_type():
    listing = {
        "source": "statvin",
        "seller_type": "private",  # should be irrelevant — source wins
        "description": "",
    }
    assert classify_listing(listing) == LISTING_TYPE_SALVAGE


def test_explicit_salvage_fields_force_salvage_classification():
    listing = {
        "source": "some_future_source",
        "seller_type": "private",
        "damage_primary": "front end",
        "description": "",
    }
    assert classify_listing(listing) == LISTING_TYPE_SALVAGE


def test_dealer_match_reason_returns_none_for_clean_private_listing():
    assert dealer_match_reason("Jane Doe", "Selling my car, one owner, no accidents") is None


def test_dealer_match_reason_is_case_insensitive():
    assert dealer_match_reason(None, "CERTIFIED PRE-OWNED, HST NOT INCLUDED") is not None
