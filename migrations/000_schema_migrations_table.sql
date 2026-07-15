-- 000_schema_migrations_table.sql
--
-- WHAT THIS CHANGES: creates the bookkeeping table the migration runner uses
-- to know which migrations have already been applied. Pure addition, no
-- existing table touched.
-- REVERSIBLE: Yes (DROP TABLE schema_migrations) — only safe once no other
-- migration has been applied, since it forgets the applied-history.
-- MODIFIES EXISTING DATA: No.

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(16) PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
