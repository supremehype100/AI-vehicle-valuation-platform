"""Regression tests for bugs fixed during the pre-publication cleanup."""
import Facebook
import config
import Model
from Features import extract_description_flags, extract_transmission
from Ingest import sold_sweep


def test_new_alone_does_not_flag_recent_service():
    assert extract_description_flags("Brand new look, new to the market")["recent_service"] == 0
    assert extract_description_flags("Full service history, dealer serviced")["recent_service"] == 1


def test_sold_sweep_with_nothing_seen_removes_nothing():
    # session=None would blow up if the sweep tried to touch the database.
    assert sold_sweep("kijiji", set(), None) == {"removed": 0}


def test_database_url_pins_psycopg2_driver():
    assert config.DATABASE_URL.startswith("postgresql+psycopg2://")


def test_valuate_uses_same_transmission_inference_as_training():
    assert Model.extract_transmission is extract_transmission
    assert extract_transmission("6 speed manual, runs great") == "manual"
    assert extract_transmission(None) == "automatic"


def _fb_page(**overrides):
    page = {
        "title": "2015 Toyota Corolla",
        "price": "$9,500",
        "mileage": "120,000 km",
        "description": "Clean car, well maintained",
        "seller_name": "Bob",
        "location": "Calgary",
    }
    page.update(overrides)
    return page


def test_facebook_listing_has_url_and_location():
    listing = Facebook.normalize_visible(_fb_page(), "123")
    assert listing["url"] == "https://www.facebook.com/marketplace/item/123"
    assert listing["location"] == "Calgary"
    assert (listing["year"], listing["make"], listing["mileage"]) == (2015, "Toyota", 120000)
    assert "fingerprint" not in listing  # computed centrally by Ingest


def test_facebook_drops_invalid_price_and_parts_cars():
    assert Facebook.normalize_visible(_fb_page(price="Contact for Price"), "1") is None
    assert Facebook.normalize_visible(_fb_page(title="2010 Honda Civic parts car"), "2") is None


def test_facebook_not_listed_mileage_is_none():
    assert Facebook.normalize_visible(_fb_page(mileage="Not Listed"), "3")["mileage"] is None
