-- 002_condition_assessments.sql
--
-- WHAT THIS CHANGES:
--   1. Creates `condition_assessments` — an insert-only audit trail of every
--      AI condition-extraction run (mirrors the existing `listing_snapshots`
--      pattern: never updated, only inserted).
--   2. Adds the "current merged condition" columns to `listings` — these are
--      the fields the ML feature pipeline and app actually read, kept in
--      sync from condition_assessments via the backfill's precedence merge.
--
-- REVERSIBLE: Yes, see DOWN section.
-- MODIFIES EXISTING DATA: No. All new columns are nullable with no default
--   that could change existing semantics; existing rows simply get NULLs
--   until the backfill (or fresh AI extraction) populates them.
-- ESTIMATED IMPACT: Fast metadata-only ADD COLUMN on Postgres 11+; the new
--   table starts empty.

BEGIN;

CREATE TABLE IF NOT EXISTS condition_assessments (
    id                  BIGSERIAL PRIMARY KEY,
    listing_id          BIGINT NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    model_version       VARCHAR(64) NOT NULL,
    schema_version      VARCHAR(16) NOT NULL,

    images_used         JSONB NOT NULL DEFAULT '[]',
    description_hash    VARCHAR(64),

    extraction_status   VARCHAR(16) NOT NULL DEFAULT 'success',
    error_message       TEXT,

    raw_extraction       JSONB NOT NULL DEFAULT '{}',
    confidence_overall   DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS ix_condition_assessment_listing_time
    ON condition_assessments (listing_id, created_at);

ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_overall VARCHAR(32);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_exterior VARCHAR(32);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_interior VARCHAR(32);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_mechanical VARCHAR(32);
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_accident_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_rust_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_flood_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_smoke_pet_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_neglect_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_recent_repair_indicated BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_undisclosed_issue_flag BOOLEAN;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_positive_notes TEXT;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_confidence DOUBLE PRECISION;
ALTER TABLE listings ADD COLUMN IF NOT EXISTS condition_field_provenance JSONB NOT NULL DEFAULT '{}';

INSERT INTO schema_migrations (version, applied_at) VALUES ('002', now())
    ON CONFLICT (version) DO NOTHING;

COMMIT;

-- DOWN (manual rollback, not auto-applied):
-- BEGIN;
-- DROP TABLE IF EXISTS condition_assessments;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_overall;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_exterior;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_interior;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_mechanical;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_accident_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_rust_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_flood_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_smoke_pet_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_neglect_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_recent_repair_indicated;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_undisclosed_issue_flag;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_positive_notes;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_confidence;
-- ALTER TABLE listings DROP COLUMN IF EXISTS condition_field_provenance;
-- DELETE FROM schema_migrations WHERE version = '002';
-- COMMIT;
