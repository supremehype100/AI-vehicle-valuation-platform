-- 004_backfill_listing_type.sql
--
-- *** THIS MIGRATION MODIFIES EXISTING DATA (an UPDATE, not just a schema
-- change). Review before running against production. ***
--
-- WHAT THIS CHANGES:
--   Sets listing_type on existing rows from the already-recorded
--   seller_type, so historical data classifies correctly under the new
--   private/dealer/salvage split instead of sitting at the 'private' column
--   default applied by migration 001.
--     seller_type = 'dealer'  -> listing_type = 'dealer'
--     source = 'statvin' (or any future auction source) -> 'salvage'
--     everything else (seller_type = 'private' or NULL) -> stays 'private'
--   This is a reclassification of EXISTING rows using data you already
--   collected — it does not invent new information, and today's scrapers
--   (Facebook/Kijiji/AutoTrader) only ever wrote 'private' or 'dealer' rows,
--   so nothing currently in listings should end up classified as 'salvage'
--   until the stat.vin scraper (source='statvin') has actually run.
--
-- REVERSIBLE: Yes — listing_type can be recomputed from seller_type/source
--   at any time by re-running this script; no source data is destroyed.
-- MODIFIES EXISTING DATA: Yes (see above). No rows are deleted.
-- ESTIMATED IMPACT: One UPDATE pass over all `dealer`-seller_type rows plus
--   one over any 'statvin'-source rows — proportional to table size, a few
--   seconds to low minutes depending on row count. Runs inside a single
--   transaction so it's all-or-nothing.
--
-- RUN WITH:
--   psql "$DATABASE_URL" -f migrations/004_backfill_listing_type.sql

BEGIN;

UPDATE listings
SET listing_type = 'dealer'
WHERE seller_type = 'dealer'
  AND listing_type = 'private';

UPDATE listings
SET listing_type = 'salvage'
WHERE source = 'statvin'
  AND listing_type <> 'salvage';

-- Now that historical rows are classified, enforce the CHECK constraint
-- added (but left NOT VALID) in migration 001 against the full table.
ALTER TABLE listings VALIDATE CONSTRAINT chk_listing_type;

INSERT INTO schema_migrations (version, applied_at) VALUES ('004', now())
    ON CONFLICT (version) DO NOTHING;

COMMIT;

-- DOWN (manual rollback, not auto-applied — recomputes from source data,
-- doesn't "undo" in the traditional sense since nothing was destroyed):
-- UPDATE listings SET listing_type = 'private';
-- UPDATE listings SET listing_type = 'dealer' WHERE seller_type = 'dealer';
-- DELETE FROM schema_migrations WHERE version = '004';
