# Database guide

The project stores everything in one PostgreSQL database (default name
`vehicle_intel`). Tables are defined in `models.py`; extra constraints, indexes
and views come from the numbered files in `migrations/`.

## Setting it up

```bash
python run.py init_db          # create tables, then apply migrations (safe to repeat)
python run.py status           # row counts
python migrations/migrate.py --status    # which migrations are applied
```

`init_db` runs `Session.create_tables()` (SQLAlchemy creates whatever is
missing) followed by the migration runner. Run it again after pulling new
code; already-applied migrations are skipped.

## Tables

### `listings`: current state of every ad
One row per (`source`, `source_id`), enforced by `uq_listing_source_id`.

| Group | Columns |
| --- | --- |
| Identity | `id`, `source`, `source_id`, `url`, `vin` (indexed), `fingerprint` (indexed) |
| Vehicle | `title`, `make`, `model`, `year`, `mileage`, `trim`, `color` (composite index on make/model/year) |
| Listing | `price`, `description`, `status` (`active` or `removed`), `seller_type`, `seller_id`, `location` |
| Timing | `first_seen`, `last_seen`, `posted_at`, `days_on_market` |
| Media / raw | `image_urls` (text array), `raw_json` (JSONB with transmission, fuel type, body type, drivetrain and the original record) |
| Classification | `listing_type`: `private`, `dealer` or `salvage` (indexed, checked by constraint) |
| Salvage only | `damage_primary`, `damage_secondary`, `title_type`, `auction_house`, `auction_date`, `lot_number`, `run_drive_status`, `auction_sale_status` |
| AI condition | `condition_overall`, `_exterior`, `_interior`, `_mechanical`, `condition_*_indicated` booleans (accident, rust, flood, smoke/pet, neglect, recent repair), `condition_undisclosed_issue_flag`, `condition_positive_notes`, `condition_confidence`, `condition_field_provenance` (JSONB: which tier set each field) |

`NULL` in a boolean condition column means *no evidence either way*, which is
different from `false` (*evidence of absence*).

### `listing_snapshots`: price and status history
Insert-only. A row is written when a listing is first seen, whenever its price
or status changes, and when it is marked removed. Use it for price-drop
analysis and time-to-sell.

### `condition_assessments`: AI audit trail
Insert-only. One row per extraction attempt: `model_version`, `schema_version`,
`images_used`, `description_hash`, `extraction_status` (`success`/`failed`),
`error_message`, `raw_extraction` (JSONB), `confidence_overall`.

### `model_artifacts`: model registry
One row per trained model: `model_key` (`private_csv`, `private_traintest`,
`dealer_traintest`, `salvage_traintest`), `version` (UTC timestamp, unique with
the key), file paths, dataset row count and fingerprint, split strategy,
feature list, config and a `metrics` JSONB. NaN/infinite metrics are stored as
`null` because JSONB cannot hold them.

### `training_runs` and `price_predictions`
`training_runs` logs each baseline training (`Model.train`): sample counts,
MAE, R², median % error, whether a personal CSV was used.
`price_predictions` stores per-listing valuations written by
`Model.score_listing`.

### `schema_migrations`
Bookkeeping for the migration runner (one row per applied version).

## Views

Created by migration `005`. They are plain filters on `listings`, handy for
ad-hoc SQL so you cannot forget the type filter.

```sql
SELECT count(*) FROM private_listings;
SELECT count(*) FROM dealer_listings;
SELECT count(*) FROM salvage_listings;
```

## Migrations

Numbered `.sql` files in `migrations/`, applied in order by
`python migrations/migrate.py`.

| # | File | What it does | Changes existing data? |
| --- | --- | --- | --- |
| 000 | `schema_migrations_table` | Bookkeeping table | No |
| 001 | `listing_type_and_salvage_fields` | `listing_type` column + constraint + index, salvage columns | No (adds defaults) |
| 002 | `condition_assessments` | Audit table and `condition_*` columns | No |
| 003 | `model_registry` | `model_artifacts` table | No |
| 004 | `backfill_listing_type` | Sets `listing_type` from `seller_type`, validates the constraint | **Yes**, recomputable, nothing deleted |
| 005 | `dataset_views` | The three views | No |

Every file starts with a header explaining what it changes, whether it is
reversible and how to roll it back. The `DOWN` section at the bottom of each
file is commented out and is only ever run by hand.

Commands:

```bash
python migrations/migrate.py --status      # applied vs pending
python migrations/migrate.py --dry-run     # list what would run
python migrations/migrate.py               # apply
psql "$DATABASE_URL" -f migrations/004_backfill_listing_type.sql   # run one file directly
```

To add a migration, create the next numbered file (for example
`006_add_something.sql`), keep it idempotent (`IF NOT EXISTS`) and document it
in the same header style.

## Handy queries

```sql
-- Listings per source and type
SELECT source, listing_type, count(*) FROM listings GROUP BY 1, 2 ORDER BY 3 DESC;

-- Median asking price by make for active private listings
SELECT make, count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
FROM private_listings WHERE status = 'active' GROUP BY make HAVING count(*) >= 20
ORDER BY median_price DESC;

-- How long do listings last before disappearing?
SELECT make, round(avg(days_on_market)) AS avg_days, count(*)
FROM listings WHERE status = 'removed' AND days_on_market IS NOT NULL
GROUP BY make HAVING count(*) >= 10 ORDER BY avg_days;

-- Listings whose price dropped at least once
SELECT l.id, l.title, max(s.price) AS highest, min(s.price) AS lowest
FROM listings l JOIN listing_snapshots s ON s.listing_id = l.id
GROUP BY l.id, l.title HAVING max(s.price) > min(s.price);

-- AI condition coverage
SELECT count(*) FILTER (WHERE condition_confidence IS NOT NULL) AS assessed, count(*) AS total
FROM listings;

-- Latest registered model per key
SELECT DISTINCT ON (model_key) model_key, version, metrics->>'mae' AS mae, trained_at
FROM model_artifacts ORDER BY model_key, trained_at DESC;
```

## Backups

```bash
# dump (works for Docker too: docker compose exec db pg_dump -U postgres vehicle_intel > backup.sql)
pg_dump "$DATABASE_URL" -Fc -f vehicle_intel.dump

# restore into an empty database
pg_restore -d vehicle_intel --clean vehicle_intel.dump
```

Trained model files in `MODEL_DIR` are **not** in the database dump; back that
folder up too, or retrain. Photos in `image_cache/` are a cache and can be
discarded.

## Resetting

To start over completely (this deletes all collected data):

```bash
docker compose down -v        # also removes the db_data and ollama_data volumes
```

To empty only the listings but keep the schema:

```sql
TRUNCATE listings RESTART IDENTITY CASCADE;
```
