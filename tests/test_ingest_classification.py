"""
Tests that Ingest.upsert() assigns listing_type at insert time via
classify_listing(), and honors an explicit listing_type if the scraper
already provided one (e.g. scrapers/statvin.py setting listing_type="salvage"
directly rather than relying on inference).

Listing/DBListingSnapshot use Postgres-only column types (JSONB, ARRAY), so a
real SQLite/in-memory DB round-trip isn't practical here — instead we use the
REAL Listing ORM class (plain Python object construction needs no DB) with a
mocked Session, and inspect the object actually passed to session.add().
"""
from unittest.mock import MagicMock

import Ingest


def _make_mock_session_with_no_existing_row():
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    return session


def _added_listing(session):
    """The Listing instance passed to the first session.add() call."""
    first_call_args = session.add.call_args_list[0]
    return first_call_args[0][0]


def test_new_private_listing_gets_private_listing_type():
    session = _make_mock_session_with_no_existing_row()
    listing_dict = {
        "source": "kijiji",
        "source_id": "abc123",
        "seller_type": "private",
        "description": "clean car",
        "make": "Honda",
        "model": "Civic",
    }
    Ingest.upsert(listing_dict, session)
    assert _added_listing(session).listing_type == "private"


def test_new_dealer_listing_gets_dealer_listing_type():
    session = _make_mock_session_with_no_existing_row()
    listing_dict = {
        "source": "kijiji",
        "source_id": "abc124",
        "seller_type": "dealer",
        "description": "",
        "make": "Honda",
        "model": "Civic",
    }
    Ingest.upsert(listing_dict, session)
    assert _added_listing(session).listing_type == "dealer"


def test_explicit_listing_type_overrides_inference():
    session = _make_mock_session_with_no_existing_row()
    listing_dict = {
        "source": "statvin",
        "source_id": "lot999",
        "seller_type": None,
        "listing_type": "salvage",
        "description": "",
    }
    Ingest.upsert(listing_dict, session)
    assert _added_listing(session).listing_type == "salvage"


def test_statvin_source_infers_salvage_even_without_explicit_field():
    session = _make_mock_session_with_no_existing_row()
    listing_dict = {
        "source": "statvin",
        "source_id": "lot1000",
        "seller_type": None,
        "description": "",
    }
    Ingest.upsert(listing_dict, session)
    assert _added_listing(session).listing_type == "salvage"
