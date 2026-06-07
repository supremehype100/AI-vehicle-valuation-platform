from __future__ import annotations
"""
ml/features.py

Converts raw listing data into features for the price prediction model.

Features used:
  - year, mileage                          (numeric)
  - make, model, trim                      (categorical → label encoded)
  - seller_type (dealer vs private)        (binary)
  - days_on_market                         (numeric)
  - transmission                           (manual vs automatic — extracted from description)
  - body_type, drivetrain, fuel_type       (categorical, from raw_json)
  - description signals                    (binary flags from NLP)
"""

import re
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select


# ---------------------------------------------------------------------------
# Transmission extraction
# Sellers almost always mention "manual" explicitly — silence means automatic
# ---------------------------------------------------------------------------

def extract_transmission(description: str | None, raw_transmission: str | None = None) -> str:
    """
    Determine transmission type.
    Priority:
      1. Structured attribute from scraper (e.g. Kijiji cartransmission field)
      2. Keyword scan of description text
      3. Default: "automatic"
    """
    # Kijiji cartransmission returns "1" for manual, "2" for automatic
    if raw_transmission:
        t = str(raw_transmission).lower().strip()
        if t in ("manual", "1", "m", "mt", "5-speed", "6-speed"):
            return "manual"
        if t in ("automatic", "2", "a", "at", "auto"):
            return "automatic"

    # Fall back to description scan
    if description:
        if re.search(r'\bmanual\b', description, re.IGNORECASE):
            return "manual"

    return "automatic"


# ---------------------------------------------------------------------------
# Description keyword signals
# These turn free-text descriptions into binary ML features
# ---------------------------------------------------------------------------

POSITIVE_SIGNALS = {
    "accident_free":  r"accident[- ]free|no accidents?|0 accidents?|clean carfax",
    "one_owner":      r"one owner|1 owner|single owner|first owner|original owner",
    "new_tires":      r"new tires?|winter tires?|all[- ]season",
    "new_brakes":     r"new brakes?|brakes? replaced",
    "recent_service": r"recently serviced|just serviced|full service|dealer serviced|full service history|dealer service history|service history|new",
    "no_rust":        r"no rust|rust free|rust-free",
    "immaculate":     r"immaculate|mint|pristine",
    "rebuilt_engine": r"rebuilt engine|new engine|engine rebuilt",
    "rebuilt_transmission": r"rebuilt transmission|new transmission",
}

NEGATIVE_SIGNALS = {
    "rebuilt":        r"rebuilt|recertified|rebuilt title",
    "salvage":        r"salvage|insurance write[- ]off",
    "needs_work":     r"needs (work|repair|tlc|brakes?|tires?|engine|transmission|timing|paint|body work)|as[- ]is|sold as[- ]is",
    "rust":           r"\brust(y|ed)?\b",
    "inop":           r"\binop\b|not (working|functional)|doesn.t work|broken",
    "leak":           r"\bleak(s|ing)?\b",
    "hail":           r"\bhail\b",
    "misfire":        r"\bmisfire\b",
}


def extract_description_flags(description: str | None) -> dict[str, int]:
    if not description:
        return {k: 0 for k in {**POSITIVE_SIGNALS, **NEGATIVE_SIGNALS}}
    text = description.lower()
    flags = {}
    for name, pattern in POSITIVE_SIGNALS.items():
        flags[name] = int(bool(re.search(pattern, text)))
    for name, pattern in NEGATIVE_SIGNALS.items():
        flags[name] = int(bool(re.search(pattern, text)))
    return flags


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build feature DataFrame from DB listings
# ---------------------------------------------------------------------------

def listings_to_dataframe(listings) -> pd.DataFrame:
    rows = []
    for l in listings:
        if l.price is None or l.year is None:
            continue

        rj = l.raw_json or {}
        desc_flags = extract_description_flags(l.description)

        row = {
            "price": l.price,
            "year": l.year,
            "mileage": l.mileage if l.mileage is not None else 100_000,
            "days_on_market": l.days_on_market or 0,
            "make": (l.make or "unknown").lower().strip(),
            "model": (l.model or "unknown").lower().strip(),
            "trim": (l.trim or "unknown").lower().strip(),
            "seller_type": l.seller_type or "private",
            "body_type": (rj.get("body_type") or "unknown").lower().strip(),
            "drivetrain": (rj.get("drivetrain") or "unknown").lower().strip(),
            "fuel_type": (rj.get("fuel_type") or "unknown").lower().strip(),

            # Transmission: structured attribute first, then description scan
            "transmission": extract_transmission(l.description, rj.get("transmission")),

            **desc_flags,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_training_data(session: Session, min_price: float = 500, max_price: float = 200_000) -> pd.DataFrame:
    from models import Listing
    listings = session.execute(
        select(Listing).where(
            Listing.price >= min_price,
            Listing.price <= max_price,
            Listing.year.isnot(None),
        )
    ).scalars().all()
    return listings_to_dataframe(listings)


def prepare_features(df: pd.DataFrame, label_encoders: dict | None = None, fit: bool = False):
    """
    Encode categorical columns.
    fit=True  → create new encoders (training time)
    fit=False → use provided encoders (inference time)
    Returns (X, y, encoders)
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    target = df.pop("price")

    CAT_COLS = [
        "make", "model", "trim", "seller_type",
        "body_type", "drivetrain", "fuel_type", "transmission",
    ]

    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "unknown"
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "unknown")
            if "unknown" not in known:
                le.classes_ = np.append(le.classes_, "unknown")
            df[col] = le.transform(df[col].astype(str))

    return df.values, target.values, label_encoders