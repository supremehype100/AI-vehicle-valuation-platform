"""
migrations/migrate.py — resumable SQL migration runner.

This project didn't use Alembic before, and introducing it wholesale for one
batch of changes would be a bigger footprint than the problem needs. Instead:
plain numbered .sql files (000, 001, 002, ...), each wrapped in its own
transaction, tracked in a `schema_migrations` table. Safe to re-run — already
-applied migrations are skipped.

Usage:
    python migrations/migrate.py                 # apply all pending migrations
    python migrations/migrate.py --dry-run        # show what WOULD run
    python migrations/migrate.py --status          # show applied vs pending

Every migration file documents, in its header comment, whether it modifies
existing data and whether it's reversible — read that before running against
production. This script only ever runs migrations forward; rollbacks are the
commented-out DOWN section at the bottom of each file, run manually.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_URL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def _applied_versions(engine) -> set[str]:
    with engine.connect() as conn:
        exists = conn.execute(sa.text(
            "SELECT to_regclass('public.schema_migrations') IS NOT NULL"
        )).scalar()
        if not exists:
            return set()
        rows = conn.execute(sa.text("SELECT version FROM schema_migrations")).fetchall()
        return {r[0] for r in rows}


def run(dry_run: bool = False, status_only: bool = False) -> None:
    engine = sa.create_engine(DATABASE_URL)
    files = _migration_files()
    if not files:
        logger.info("No migration files found in %s", MIGRATIONS_DIR)
        return

    applied = _applied_versions(engine)
    pending = [f for f in files if f.name.split("_", 1)[0] not in applied]

    if status_only:
        for f in files:
            version = f.name.split("_", 1)[0]
            mark = "applied" if version in applied else "PENDING"
            print(f"  [{mark:7}] {f.name}")
        return

    if not pending:
        logger.info("Database is up to date (%d migrations already applied).", len(applied))
        return

    logger.info("%d pending migration(s): %s", len(pending), ", ".join(f.name for f in pending))

    if dry_run:
        logger.info("--dry-run: not applying anything.")
        return

    for f in pending:
        version = f.name.split("_", 1)[0]
        sql = f.read_text(encoding="utf-8")
        logger.info("Applying %s ...", f.name)
        try:
            with engine.begin() as conn:
                # The .sql files each wrap themselves in BEGIN/COMMIT for the
                # psql-direct use case; executed via SQLAlchemy we let the
                # engine own the transaction instead, so strip those lines to
                # avoid nested transactions.
                statements = _strip_bare_transaction_control(sql)
                for stmt in statements:
                    conn.execute(sa.text(stmt))
        except Exception:
            logger.exception("Migration %s FAILED — stopping. Fix the issue and re-run; "
                              "already-applied migrations will be skipped.", f.name)
            raise
        logger.info("Applied %s", f.name)

    logger.info("Done. %d migration(s) applied this run.", len(pending))


def _strip_bare_transaction_control(sql: str) -> list[str]:
    """Split a migration file into statements, dropping BEGIN/COMMIT lines
    and SQL comments so it can run inside a SQLAlchemy-managed transaction."""
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        upper = stripped.upper()
        if upper in ("BEGIN;", "COMMIT;", "BEGIN", "COMMIT"):
            continue
        lines.append(line)
    joined = "\n".join(lines)
    # naive split on ';' at end-of-statement — fine here because none of our
    # migrations contain semicolons inside string literals or function bodies.
    statements = [s.strip() for s in joined.split(";")]
    return [s + ";" for s in statements if s]


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    status_only = "--status" in sys.argv
    run(dry_run=dry_run, status_only=status_only)
