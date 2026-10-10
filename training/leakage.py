from __future__ import annotations

"""
training/leakage.py

Documentation-as-code: for every column `Features.listings_to_dataframe()`
currently produces, is it genuinely available at prediction time (i.e. known
about a vehicle BEFORE you'd ask the model for a price), and why/why not.

THE KNOWN LEAK: days_on_market
-------------------------------
`Features.listings_to_dataframe()` sets `days_on_market` from each listing's
CURRENT `days_on_market` value (`l.days_on_market or 0`). For a listing that
has since sold or been removed, that's the FINAL days-on-market — a number
that is only knowable once the listing is already gone. At the moment you'd
actually want a price prediction (a fresh listing, or a hypothetical
vehicle), you cannot know how many days it will sit on the market — that's
downstream of (and correlated with) the very price decision the model is
supposed to inform.

Yet `Model.valuate()` always calls with `days_on_market=0` at inference time
(see its `row = {...}` dict — `"days_on_market": 0` is hardcoded). Training on
the real, nonzero values and serving with 0 is a train/serve skew that lets a
model learn spurious structure from days_on_market that it can never use
correctly at inference.

`zero_out_leaky_features()` below forces `days_on_market = 0` on every row
(train AND test) BEFORE `Features.prepare_features()` is called, so the
column becomes a constant the model cannot extract any predictive signal
from — matching its always-0 value at inference. Every pipeline uses it:
`Model.train()` (Model A / private_csv) as well as private_traintest,
dealer_traintest and salvage_traintest.

FEATURE_AVAILABILITY
---------------------
Keyed by every column name that `Features.listings_to_dataframe()` produces:
`price` (the target, included for completeness) plus every categorical/
numeric feature and every description-derived boolean flag from
`extract_description_flags()`. Each entry is
`{"available_at_prediction_time": bool, "reason": str}`.
"""

# Names of the boolean description-flag columns, as produced by
# Features.extract_description_flags() (POSITIVE_SIGNALS + NEGATIVE_SIGNALS).
_DESCRIPTION_FLAG_REASON_AVAILABLE = (
    "Derived purely from listing description text via regex, which is "
    "supplied by the caller (or scraped) before a prediction is requested. "
    "Defaults to 0 when no description is given, same as how training rows "
    "with no description behave — no train/serve skew."
)

FEATURE_AVAILABILITY: dict[str, dict[str, object]] = {
    "price": {
        "available_at_prediction_time": True,
        "reason": (
            "This is the TARGET, not a predictive feature. Always known for "
            "historical training rows (it's what was actually sold/listed "
            "for); trivially 'unavailable' for the vehicle being priced, "
            "which is the entire point of the model. Listed here only for "
            "completeness since listings_to_dataframe() does produce it."
        ),
    },
    "year": {
        "available_at_prediction_time": True,
        "reason": "Model year is known for any real or hypothetical vehicle up front.",
    },
    "mileage": {
        "available_at_prediction_time": True,
        "reason": "Supplied by the user/caller at valuation time (Model.valuate() takes it as a required arg).",
    },
    "days_on_market": {
        "available_at_prediction_time": False,
        "reason": (
            "LEAKY — see module docstring. Trains on each listing's CURRENT "
            "(often final, post-hoc) days-on-market value, but "
            "Model.valuate() always passes 0 at inference. Neutralized in "
            "every *_traintest pipeline via zero_out_leaky_features()."
        ),
    },
    "make": {
        "available_at_prediction_time": True,
        "reason": "Known for any real or hypothetical vehicle up front.",
    },
    "model": {
        "available_at_prediction_time": True,
        "reason": "Known for any real or hypothetical vehicle up front.",
    },
    "trim": {
        "available_at_prediction_time": True,
        "reason": "Supplied by caller when known; defaults to 'unknown' identically at train and serve time.",
    },
    "seller_type": {
        "available_at_prediction_time": True,
        "reason": "Determined by which pipeline/dataset is being used (private/dealer), known before prediction.",
    },
    "body_type": {
        "available_at_prediction_time": True,
        "reason": "Vehicle attribute, supplied by caller or looked up from raw_json at listing time; known up front.",
    },
    "drivetrain": {
        "available_at_prediction_time": True,
        "reason": "Vehicle attribute, known up front; defaults to 'unknown' identically at train and serve time.",
    },
    "fuel_type": {
        "available_at_prediction_time": True,
        "reason": "Vehicle attribute, known up front; defaults to 'unknown' identically at train and serve time.",
    },
    "transmission": {
        "available_at_prediction_time": True,
        "reason": (
            "Inferred from description/raw_json text at training time; at "
            "serve time Model.valuate() applies the same description-based "
            "inference (extract_transmission), defaulting to 'automatic' "
            "exactly as training does — no train/serve skew, and no future "
            "information used."
        ),
    },
    # --- description-derived boolean flags ---
    "accident_free": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "one_owner": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "new_tires": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "new_brakes": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "recent_service": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "no_rust": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "immaculate": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "rebuilt_engine": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "rebuilt_transmission": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "rebuilt": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "salvage": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "needs_work": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "rust": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "leak": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "hail": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "misfire": {"available_at_prediction_time": True, "reason": _DESCRIPTION_FLAG_REASON_AVAILABLE},
    "inop": {
        "available_at_prediction_time": True,
        "reason": (
            _DESCRIPTION_FLAG_REASON_AVAILABLE
            + " Note: listings_to_dataframe() actually FILTERS OUT rows "
            "where this flag is 1 (parts_car/inop listings are dropped "
            "entirely before training), so in practice this column is "
            "always 0 in the training data — not a leak, just a "
            "near-constant column worth knowing about."
        ),
    },
    "parts_car": {
        "available_at_prediction_time": True,
        "reason": (
            _DESCRIPTION_FLAG_REASON_AVAILABLE
            + " Note: same filtering as 'inop' above — rows with this flag "
            "set are dropped before training, so it's always 0 in-sample."
        ),
    },
}


def zero_out_leaky_features(df):
    """
    Force days_on_market = 0 for every row (train AND test) before
    Features.prepare_features() is called. See module docstring for why:
    the column is leaky (trains on real/final days-on-market, but
    Model.valuate() always serves with 0), so we neutralize it into a
    constant the model can't extract spurious signal from, matching its
    always-0 value at inference.

    No-ops (returns df unchanged) if the column isn't present, so this is
    safe to call unconditionally.
    """
    df = df.copy()
    if "days_on_market" in df.columns:
        df["days_on_market"] = 0
    return df
