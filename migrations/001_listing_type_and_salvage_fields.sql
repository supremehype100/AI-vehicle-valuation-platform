-- 001_listing_type_and_salvage_fields.sql
--
-- WHAT THIS CHANGES:
--   Adds the explicit listing_type discriminator ('private' | 'dealer' |
--   'salvage') plus nullable salvage-auction columns to the existing
--   `listings` table.
--
-- REVERSIBLE: Yes — see the DOWN section at the bottom (commented out).
--   Dropping columns loses any data written to them since they were added,
--   but does not touch any pre-existing column.
--
-- MODIFIES EXISTING DATA: No destructive changes. `listing_type` is added
--   with a DEFAULT so every existing row gets a value; no existing column is
--   altered, dropped, or renamed. Nothing is deleted.
--
-- ESTIMATED IMPACT: A few seconds on tables up to several million rows —
--   these are simple ADD COLUMN ... DEFAULT operations (fast in Postgres 11+,
--   which stores the default as metadata rather than rewriting the table).
--
-- RUN WITH:
--   psql "$DATABASE_URL" -f migrations/001_listing_type_and_salvage_fields.sql
--   (or: python migrations/migrate.py — see that file for the resumable runner)

BEGIN;

ALTER TABLE listings
    ADD COLUMN IF NOT EXISTS listing_type VARCHAR(16) NOT NULL DEFAULT 'private';

ALTER TABLE listings
    ADD CONSTRAINT chk_listing_type
    CHECK (listing_type IN ('private', 'dealer', 'salvage')) NOT VALID;
-- NOT VALID = enforced for new/updated rows immediately, existing rows
-- validated lazily via a separate VALIDATE CONSTRAINT so this migration
-- never locks the table for a full scan. Run once the data backfill
-- (004_backfill_listing_type.sql) has completed:
--   ALTER TABLE listings VALIDATE CONSTRAINT chk_listing_type;

CREATE INDEX IF NOT EXISTS ix_listing_type ON listings (listing_type);

ALTER TABLE listings ADD COLUMN IF NOT EXISTS damage_primary VARCHAR(64);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS damage_secondary VARCHAR(64);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS title_type VARCHAR(64);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS auction_house VARCHAR(64);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS auction_date TIMESTAMPTZ;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS lot_number VARCHAR(64);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS run_drive_status VARCHAR(32);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS auction_sale_status VARCHAR(16);

INSERT INTO schema_migrations (version, applied_at) VALUES ('001', now())
    ON CONFLICT (version) DO NOTHING;

COMMIT;

-- DOWN (manual rollback, not auto-applied):
-- BEGIN;
-- ALTER TABLE listings DROP CONSTRAINT IF EXISTS chk_listing_type;
-- DROP INDEX IF EXISTS ix_listing_type;
-- ALTER TABLE listings DROP COLUMN IF EXISTS listing_type;
-- ALTER TABLE listings DROP COLUMN IF EXISTS damage_primary;
-- ALTER TABLE listings DROP COLUMN IF EXISTS damage_secondary;
-- ALTER TABLE listings DROP COLUMN IF EXISTS title_type;
-- ALTER TABLE listings DROP COLUMN IF EXISTS auction_house;
-- ALTER TABLE listings DROP COLUMN IF EXISTS auction_date;
-- ALTER TABLE listings DROP COLUMN IF EXISTS lot_number;
-- ALTER TABLE listings DROP COLUMN IF EXISTS run_drive_status;
-- ALTER TABLE listings DROP COLUMN IF EXISTS auction_sale_status;
-- DELETE FROM schema_migrations WHERE version = '001';
-- COMMIT;
