# How the price models work

This page explains what is predicted, from what, how honesty is built into
the evaluation, and how to read the results.

## What is predicted

The target is a listing's **asking price** in dollars. That is what sellers ask,
not necessarily what buyers pay. It is a good proxy for the market, but expect
asking prices to run somewhat above final sale prices.

## What the models use today

For the three retail models (1, 2, 3), each listing becomes one row with:

| Feature | Type | Source |
| --- | --- | --- |
| `year`, `mileage` | numbers | listing (missing mileage becomes 100,000) |
| `make`, `model`, `trim` | text, label-encoded | listing |
| `seller_type` | text, label-encoded | listing |
| `body_type`, `drivetrain`, `fuel_type` | text, label-encoded | `raw_json` extras, `unknown` if absent |
| `transmission` | `manual` / `automatic` | description text (default automatic) |
| 9 positive flags | 0/1 | description regexes: `accident_free`, `one_owner`, `new_tires`, `new_brakes`, `recent_service`, `no_rust`, `immaculate`, `rebuilt_engine`, `rebuilt_transmission` |
| 9 negative flags | 0/1 | `rebuilt`, `salvage`, `needs_work`, `rust`, `inop`, `leak`, `hail`, `misfire`, `parts_car` |
| `days_on_market` | number | **forced to 0**, see [leakage](#the-leak-that-was-found-and-neutralized) |

Rows are dropped when the price is outside $1,000-$200,000 or a placeholder,
the year is missing, or the description marks the car as a parts car or
non-running.

**The AI condition fields are not model inputs yet.** They are collected and
stored (see [MODULES.md](MODULES.md#ai-condition-extraction)) so that a future
version can bucket them into features. Adding them would be a natural next
step: see [EXTENDING.md](EXTENDING.md#add-a-feature).

**Model 4 (salvage)** uses its own features: `year`, `make`, `model`,
`mileage`, `damage_primary`, `damage_secondary`, `title_type`,
`run_drive_status`, `auction_house`. It trains only on lots whose
`auction_sale_status` is `sold`. The stat.vin scraper currently records
`pending`, so Model 4 has no rows to learn from until sold results are captured.

## The algorithm

All four use **XGBoost** (gradient-boosted decision trees), a strong default for
tabular data:

```
500 trees, learning rate 0.05, max depth 6, 80% row and column sampling,
min child weight 5, L1 0.1, L2 1.0, early stopping after 30 rounds without
improvement, seed 17
```

## Why four models and two evaluation styles

| Model | Split used for scoring |
| --- | --- |
| 1 `private_csv` | Your own CSV if `vehicles_cleaned.csv` exists, otherwise a **random** 15% |
| 2, 3, 4 | **Time-aware, VIN-disjoint, de-duplicated** newest 15% |

A random split is the textbook default and is wrong for listings. The
`training/splits.py` docstring has the full reasoning; in short:

1. **Reposts.** The same car is relisted. A random split puts one copy in
   training and one in the test set, so the model "recognises" it. De-duplicating
   by fingerprint removes this.
2. **VIN identity.** The same VIN in both sets lets the model memorise a price.
   `enforce_vin_disjoint` moves any VIN found on both sides into training and
   `assert_no_vin_leakage` checks again right before training.
3. **Time.** Prices drift. The honest question is "given the past, can I price a
   car listed *now*?", so the newest listings form the test set.
4. **Clean model selection.** Early stopping needs a validation set. Using the
   test set for it would leak test information, so a further time-based slice is
   carved from the training data only. The test set is scored exactly once.

The pipeline order:

```
dedupe_by_fingerprint → time_aware_split → enforce_vin_disjoint → assert_no_vin_leakage
  → carve an inner validation slice (same steps) → fit encoders on train only
  → train with early stopping → score the untouched test set once → save → register
```

## The leak that was found and neutralized

`days_on_market` is only known *after* a listing has sold or vanished, yet a fresh
listing being priced has no such number. Early code trained on the real values
but always predicted with 0, so the model learned a pattern it could never use.
Every pipeline now sets the column to 0 in training and testing
(`training/leakage.py`), and `Model.valuate` uses 0, so training and prediction
agree. `training/leakage.py` also documents, column by column, whether each
feature is available at prediction time.

## Reading the metrics

`compute_metrics` returns:

| Metric | Meaning | Good looks like |
| --- | --- | --- |
| `mae` | Average dollars off | Small relative to typical prices |
| `rmse` | Like MAE but punishes big misses | Close to MAE means few huge errors |
| `r2` | Share of price variation explained | Higher is better; `null` if the test prices are all identical |
| `median_abs_error` | The typical miss, less swayed by outliers | Lower than MAE if there are outliers |
| `mape` | Average % error | |
| `bias` | Mean(prediction − actual) | Near 0; positive means over-predicting |
| `pct_within_500`, `pct_within_1000`, `pct_within_2500`, `pct_within_5000` | Fraction of predictions inside ±$ thresholds | Higher is better |
| `error_distribution` | p10, p25, p50, p75, p90 of absolute error | Shows the tail |

Compare models side by side:

```bash
python run.py compare_models           # table from each model's own stored metrics → reports/model_comparison.md
python run.py compare_private_fair     # Models 1 and 2 scored on the SAME held-out rows
```

`compare_models` tabulates numbers from *different* test sets, so use it as a
summary. `compare_private_fair` is the apples-to-apples view, with the caveat
documented in `training/evaluate.py`.

## The valuation answer

`python run.py value` (or `Model.valuate`) returns:

- **Predicted price** from the baseline model (Model 1 files).
- **Likely range:** ±15% around the prediction; if at least three comparable
  listings exist, the range is clipped to their 15th-85th percentile. This is a
  heuristic, not a statistical confidence interval. A proper quantile model is a
  known follow-up.
- **Comparables:** up to five *active* listings with the same make and model,
  within two model years and 30,000 km, ranked by closeness.
- **Model MAE** and training sample count, so you can judge how much to trust it.

## Versioning and reproducibility

Every `train_*` run of Models 2-4 saves files under
`MODEL_DIR/<model_key>/<version>_{model,encoders,meta}.pkl` and inserts a
`model_artifacts` row containing the dataset fingerprint (a hash of the exact
listing ids used), the split strategy, feature list, hyper-parameters and metrics.
Failed runs register nothing, so retrying is always safe.

```python
from Session import get_session
from training import registry

with get_session() as session:
    latest = registry.get_latest(session, "private_traintest")
    print(latest.version, latest.metrics["mae"])
    for v in registry.list_versions(session, "private_traintest"):
        print(v.version, v.dataset_row_count)
```

Model 1 overwrites `xgb_price_model.pkl` in place (it predates the registry) but
also registers a row so it shows up in comparisons.

## Getting better predictions

1. **More data** from more pages and more days. Model quality tracks sample size.
2. **Keep running the sold sweep** so `days_on_market` and removal data stay accurate.
3. **Add your own trusted CSV** as `vehicles_cleaned.csv` to validate Model 1 against known good prices.
4. **Run the condition backfill**, then add condition features (see [EXTENDING.md](EXTENDING.md#add-a-feature)).
5. **Narrow the scope.** One city and a few common makes beat a whole country of everything.

## Limits to keep in mind

- Asking price is not sale price.
- The model sees only what the listing text exposes. It cannot see a dent, a smell or a service record.
- Rare models and trims have too few examples; they fall into the `unknown` bucket and the prediction regresses towards the average.
- A market shock (a sharp rate change, a recall) makes older training data stale. Retrain regularly; the scheduler does this nightly.
