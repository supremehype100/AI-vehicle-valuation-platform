-- 005_dataset_views.sql
--
-- WHAT THIS CHANGES: creates three read-only views over `listings`, one per
-- dataset, so "which table do I query for the dealer model" has an
-- unambiguous, self-documenting answer instead of everyone re-typing
-- `WHERE listing_type = 'dealer'` (and risking a typo that leaks data across
-- datasets). Storage stays unified in one physical table — see
-- training/datasets.py in the app code for why (avoids duplicating
-- snapshots/predictions/condition_assessments three ways).
-- REVERSIBLE: Yes (DROP VIEW ...).
-- MODIFIES EXISTING DATA: No. Views only.

BEGIN;

CREATE OR REPLACE VIEW private_listings AS
    SELECT * FROM listings WHERE listing_type = 'private';

CREATE OR REPLACE VIEW dealer_listings AS
    SELECT * FROM listings WHERE listing_type = 'dealer';

CREATE OR REPLACE VIEW salvage_listings AS
    SELECT * FROM listings WHERE listing_type = 'salvage';

INSERT INTO schema_migrations (version, applied_at) VALUES ('005', now())
    ON CONFLICT (version) DO NOTHING;

COMMIT;

-- DOWN:
-- BEGIN;
-- DROP VIEW IF EXISTS private_listings;
-- DROP VIEW IF EXISTS dealer_listings;
-- DROP VIEW IF EXISTS salvage_listings;
-- DELETE FROM schema_migrations WHERE version = '005';
-- COMMIT;
