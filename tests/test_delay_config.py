"""
Tests that the Facebook inter-listing delay is (a) reduced to 75% of its
original value by default and (b) actually configurable via environment
variables, without touching the separate page-load wait.
"""
import importlib
import os


ORIGINAL_MIN, ORIGINAL_MAX = 6.0, 10.0


def test_default_delay_is_75pct_of_original():
    import config
    importlib.reload(config)
    assert config.FB_LISTING_DELAY_MIN_SECONDS == ORIGINAL_MIN * 0.75
    assert config.FB_LISTING_DELAY_MAX_SECONDS == ORIGINAL_MAX * 0.75


def test_page_load_wait_is_unchanged_from_original():
    import config
    importlib.reload(config)
    assert config.FB_PAGE_LOAD_WAIT_MIN_MS == 5000
    assert config.FB_PAGE_LOAD_WAIT_MAX_MS == 7000


def test_delay_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("FB_LISTING_DELAY_MIN_SECONDS", "1.0")
    monkeypatch.setenv("FB_LISTING_DELAY_MAX_SECONDS", "2.0")
    import config
    importlib.reload(config)
    try:
        assert config.FB_LISTING_DELAY_MIN_SECONDS == 1.0
        assert config.FB_LISTING_DELAY_MAX_SECONDS == 2.0
    finally:
        monkeypatch.delenv("FB_LISTING_DELAY_MIN_SECONDS", raising=False)
        monkeypatch.delenv("FB_LISTING_DELAY_MAX_SECONDS", raising=False)
        importlib.reload(config)


def test_invalid_delay_range_raises():
    import config
    os.environ["FB_LISTING_DELAY_MIN_SECONDS"] = "10.0"
    os.environ["FB_LISTING_DELAY_MAX_SECONDS"] = "1.0"
    try:
        try:
            importlib.reload(config)
            assert False, "expected ValueError for min > max"
        except ValueError:
            pass
    finally:
        del os.environ["FB_LISTING_DELAY_MIN_SECONDS"]
        del os.environ["FB_LISTING_DELAY_MAX_SECONDS"]
        importlib.reload(config)


def test_facebook_module_actually_uses_config_values():
    """Guards against Facebook.py silently reverting to a hardcoded delay."""
    import Facebook
    import config
    assert Facebook.FB_LISTING_DELAY_MIN_SECONDS == config.FB_LISTING_DELAY_MIN_SECONDS
    assert Facebook.FB_LISTING_DELAY_MAX_SECONDS == config.FB_LISTING_DELAY_MAX_SECONDS
