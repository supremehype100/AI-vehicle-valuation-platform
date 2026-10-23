"""
Regression test for a real gap found during the audit: Features.load_training_data()
previously applied no listing_type filter at all, so dealer-tagged Kijiji rows
silently leaked into what was supposed to be the private-only training set.
This asserts the SQL filter is actually present in the query construction.
"""
from unittest.mock import MagicMock
import Features


def test_default_listing_type_filters_to_private():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = []

    Features.load_training_data(session)

    where_clause = session.execute.call_args[0][0]
    compiled = str(where_clause.compile(compile_kwargs={"literal_binds": True}))
    assert "listing_type" in compiled
    assert "'private'" in compiled


def test_explicit_listing_type_is_used():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = []

    Features.load_training_data(session, listing_type="dealer")

    where_clause = session.execute.call_args[0][0]
    compiled = str(where_clause.compile(compile_kwargs={"literal_binds": True}))
    assert "'dealer'" in compiled


def test_listing_type_none_disables_the_filter():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = []

    Features.load_training_data(session, listing_type=None)

    where_clause = session.execute.call_args[0][0]
    compiled = str(where_clause.compile(compile_kwargs={"literal_binds": True}))
    # listing_type is always present as a selected column; what must be
    # absent is a *filter* on it (an equality comparison in the WHERE
    # clause), which listing_type=None explicitly opts out of.
    assert "listing_type =" not in compiled
