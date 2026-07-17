-- 003_model_registry.sql
--
-- WHAT THIS CHANGES: creates `model_artifacts`, the version registry for all
-- four training pipelines (private_csv / private_traintest / dealer_traintest
-- / salvage_traintest).
-- REVERSIBLE: Yes (DROP TABLE model_artifacts) — loses registry history but
--   never touches the actual .pkl files on disk under MODEL_DIR.
-- MODIFIES EXISTING DATA: No. New, empty table.

BEGIN;

CREATE TABLE IF NOT EXISTS model_artifacts (
    id                      BIGSERIAL PRIMARY KEY,
    model_key               VARCHAR(32) NOT NULL,
    version                 VARCHAR(32) NOT NULL,
    is_baseline             BOOLEAN NOT NULL DEFAULT false,

    artifact_path           TEXT NOT NULL,
    encoder_path            TEXT,
    meta_path               TEXT,

    dataset_listing_type    VARCHAR(16),
    dataset_row_count       INTEGER,
    dataset_fingerprint     VARCHAR(64),
    split_strategy          VARCHAR(32),

    feature_list            JSONB NOT NULL DEFAULT '[]',
    model_config            JSONB NOT NULL DEFAULT '{}',
    metrics                 JSONB NOT NULL DEFAULT '{}',

    trained_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes                   TEXT,

    CONSTRAINT uq_model_artifact_key_version UNIQUE (model_key, version)
);

CREATE INDEX IF NOT EXISTS ix_model_artifact_key_trained
    ON model_artifacts (model_key, trained_at);

INSERT INTO schema_migrations (version, applied_at) VALUES ('003', now())
    ON CONFLICT (version) DO NOTHING;

COMMIT;

-- DOWN (manual rollback, not auto-applied):
-- BEGIN;
-- DROP TABLE IF EXISTS model_artifacts;
-- DELETE FROM schema_migrations WHERE version = '003';
-- COMMIT;
