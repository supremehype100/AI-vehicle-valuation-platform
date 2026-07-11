# models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime

class RawListing(BaseModel):
    # --- Identity ---
    source: str
    source_id: str
    url: str | None = None
    vin: str | None = None
    fingerprint: str | None = None      # hash of (year, make, model, city)

    # --- Vehicle ---
    title: str | None = None
    make: str | None = None
    model: str | None = None
    year: int | None = None
    mileage: int | None = None
    trim: str | None = None
    color: str | None = None

    # --- Listing ---
    price: float | None = None
    description: str | None = None
    status: str = "active"              # active | sold | removed | expired
    seller_type: str | None = None      # dealer | private
    seller_id: str | None = None
    location: str | None = None

    # --- Auction-specific (statvin/copart) ---
    damage_primary: str | None = None
    damage_secondary: str | None = None
    title_type: str | None = None
    auction_source: str | None = None

    # --- Tracking ---
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    posted_at: datetime | None = None

    # --- Media ---
    image_urls: list[str] = Field(default_factory=list)
    local_images: list[str] = Field(default_factory=list)

    # --- Raw ---
    raw_json: dict[str, Any] = Field(default_factory=dict)


class ListingSnapshot(BaseModel):
    """Immutable record of a listing's state at a point in time.
    Never update these — only insert. This is your price history table."""
    source: str
    source_id: str
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    price: float | None = None
    status: str | None = None
    description: str | None = None
    mileage: int | None = None          # mileage can change too (odometer corrections)



from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import hashlib


class Base(DeclarativeBase):
    pass


class Listing(Base):
    __tablename__ = "listings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    vin: Mapped[str | None] = mapped_column(String(17), index=True)
    fingerprint: Mapped[str | None] = mapped_column(String(32), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    make: Mapped[str | None] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(64), index=True)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    mileage: Mapped[int | None] = mapped_column(Integer)
    trim: Mapped[str | None] = mapped_column(String(128))
    color: Mapped[str | None] = mapped_column(String(64))
    price: Mapped[float | None] = mapped_column(Float, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active", index=True)
    seller_type: Mapped[str | None] = mapped_column(String(16))
    seller_id: Mapped[str | None] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(128))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    days_on_market: Mapped[int | None] = mapped_column(Integer)
    image_urls: Mapped[list] = mapped_column(ARRAY(Text), default=list)
    raw_json: Mapped[dict] = mapped_column(JSONB, default=dict)

    # --- Explicit dataset classification (private / dealer / salvage) ---
    # Do NOT rely on seller_type alone for training-set separation — this is
    # the field that actually gates which dataset a row belongs to. Defaults
    # to "private" so pre-existing rows (all scraped from private-only feeds
    # today) classify correctly without a data backfill being required first;
    # the migration explicitly sets it from seller_type/source for safety.
    listing_type: Mapped[str] = mapped_column(String(16), nullable=False, default="private", index=True)

    # --- Salvage-auction-specific fields (nullable; only populated for
    #     listing_type == "salvage", e.g. rows from the stat.vin scraper) ---
    damage_primary: Mapped[str | None] = mapped_column(String(64))
    damage_secondary: Mapped[str | None] = mapped_column(String(64))
    title_type: Mapped[str | None] = mapped_column(String(64))
    auction_house: Mapped[str | None] = mapped_column(String(64))
    auction_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lot_number: Mapped[str | None] = mapped_column(String(64))
    run_drive_status: Mapped[str | None] = mapped_column(String(32))
    # "sold" | "unsold" | "pending" — whether `price` is a final hammer price
    # or an estimate/current bid. Distinct from `status` (active/sold/removed
    # lifecycle) which already means something else in this codebase.
    auction_sale_status: Mapped[str | None] = mapped_column(String(16))

    # --- Current best-known condition assessment (mutable, precedence-merged
    #     from condition_assessments; mirrors how `price`/`status` above are
    #     the "current" fields while listing_snapshots is the audit trail) ---
    condition_overall: Mapped[str | None] = mapped_column(String(32))
    condition_exterior: Mapped[str | None] = mapped_column(String(32))
    condition_interior: Mapped[str | None] = mapped_column(String(32))
    condition_mechanical: Mapped[str | None] = mapped_column(String(32))
    condition_accident_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_rust_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_flood_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_smoke_pet_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_neglect_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_recent_repair_indicated: Mapped[bool | None] = mapped_column(Boolean)
    condition_undisclosed_issue_flag: Mapped[bool | None] = mapped_column(Boolean)
    condition_positive_notes: Mapped[str | None] = mapped_column(Text)
    condition_confidence: Mapped[float | None] = mapped_column(Float)
    # {field_name: {"source": "manual|structured|ai|inferred", "precedence": int,
    #   "model_version": str|None, "confidence": float|None, "set_at": iso str,
    #   "manually_validated": bool}} — per-field provenance, see config.py
    # PRECEDENCE_* constants for the merge ordering this implements.
    condition_field_provenance: Mapped[dict] = mapped_column(JSONB, default=dict)

    snapshots: Mapped[list[DBListingSnapshot]] = relationship(back_populates="listing", cascade="all, delete-orphan")
    predictions: Mapped[list[PricePrediction]] = relationship(back_populates="listing", cascade="all, delete-orphan")
    condition_history: Mapped[list["ConditionAssessment"]] = relationship(back_populates="listing", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_listing_source_id"),
        Index("ix_listing_make_model_year", "make", "model", "year"),
        Index("ix_listing_type", "listing_type"),
    )

    @staticmethod
    def make_fingerprint(year, make, model, mileage, city) -> str | None:
        if not all([year, make, model]):
            return None
        bucket = round((mileage or 0) / 5000) * 5000
        key = f"{year}:{str(make).lower()}:{str(model).lower()}:{bucket}:{str(city or '').lower()}"
        return hashlib.md5(key.encode()).hexdigest()


class DBListingSnapshot(Base):
    __tablename__ = "listing_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    listing_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    price: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(16))
    mileage: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)

    listing: Mapped[Listing] = relationship(back_populates="snapshots")

    __table_args__ = (
        Index("ix_snapshot_listing_time", "listing_id", "scraped_at"),
    )


class PricePrediction(Base):
    __tablename__ = "price_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    listing_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_low: Mapped[float | None] = mapped_column(Float)
    confidence_high: Mapped[float | None] = mapped_column(Float)
    price_delta_pct: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    shap_values: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())

    listing: Mapped[Listing] = relationship(back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("listing_id", "model_version", name="uq_prediction_listing_model"),
        Index("ix_prediction_deals", "price_delta_pct"),
    )

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now()
    )

    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    val_samples: Mapped[int] = mapped_column(Integer, nullable=False)

    mae: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    median_pct_error: Mapped[float] = mapped_column(Float, nullable=False)

    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    used_personal_csv: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )


class ConditionAssessment(Base):
    """
    Immutable audit trail of AI condition-extraction runs — insert-only, same
    pattern as DBListingSnapshot above. `Listing.condition_*` holds the
    current precedence-merged view; this table holds every attempt that fed
    into it, so a bad merge or model regression is always recoverable.
    """
    __tablename__ = "condition_assessments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    listing_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)   # e.g. "ollama:qwen2.5vl:7b"
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False)  # e.g. "condition_v1"

    # Which images were actually sent to the model, and a hash of the
    # description text used — lets the backfill detect "listing changed since
    # last assessment" without re-storing the raw payloads.
    images_used: Mapped[list] = mapped_column(JSONB, default=list)
    description_hash: Mapped[str | None] = mapped_column(String(64))

    extraction_status: Mapped[str] = mapped_column(String(16), nullable=False, default="success")  # success|failed|partial
    error_message: Mapped[str | None] = mapped_column(Text)

    # Full structured JSON returned by the model for this run — the source of
    # truth that Listing.condition_* is derived from via precedence merge.
    raw_extraction: Mapped[dict] = mapped_column(JSONB, default=dict)
    confidence_overall: Mapped[float | None] = mapped_column(Float)

    listing: Mapped[Listing] = relationship(back_populates="condition_history")

    __table_args__ = (
        Index("ix_condition_assessment_listing_time", "listing_id", "created_at"),
    )


class ModelArtifact(Base):
    """
    Model registry — one row per trained artifact version, across all four
    pipelines (private_csv / private_traintest / dealer_traintest /
    salvage_traintest). Lets `training/registry.py` answer "what's the latest
    version of model X" and "what dataset/config produced it" without
    grepping the filesystem.
    """
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_key: Mapped[str] = mapped_column(String(32), nullable=False)   # private_csv|private_traintest|dealer_traintest|salvage_traintest
    version: Mapped[str] = mapped_column(String(32), nullable=False)     # e.g. "20260812-215500"
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)    # marks the preserved Model A baseline

    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    encoder_path: Mapped[str | None] = mapped_column(Text)
    meta_path: Mapped[str | None] = mapped_column(Text)

    dataset_listing_type: Mapped[str | None] = mapped_column(String(16))  # private|dealer|salvage|null(=mixed/csv)
    dataset_row_count: Mapped[int | None] = mapped_column(Integer)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64))   # hash of row ids/version for reproducibility
    split_strategy: Mapped[str | None] = mapped_column(String(32))        # e.g. "time_aware_vin_grouped" | "personal_csv"

    feature_list: Mapped[list] = mapped_column(JSONB, default=list)
    model_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)

    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("model_key", "version", name="uq_model_artifact_key_version"),
        Index("ix_model_artifact_key_trained", "model_key", "trained_at"),
    )