"""
training/

Model training pipelines for the car-valuation project, built ON TOP OF the
existing, finalized `Features.py` / `Model.py` modules (never modifies them).

Four pipelines live here, each registering its artifacts in the
`model_artifacts` table via `training.registry`:

  - private_csv         (Model A, the untouched existing baseline — a thin
                          wrapper around Model.train())
  - private_traintest   (Model B — private listings, time-aware/VIN-disjoint
                          split instead of a random holdout)
  - dealer_traintest    (Model C — same architecture as Model B, dealer
                          listings)
  - salvage_traintest   (Model D — salvage/auction listings, custom feature
                          set; structurally complete but untested against
                          real data because no salvage rows exist yet)

Shared infrastructure:
  - training.metrics   — regression metrics used to evaluate every pipeline
  - training.splits    — leakage-aware train/test splitting utilities
  - training.leakage   — feature-availability documentation + the
                          days_on_market leak neutralizer
  - training.registry  — thin wrapper around the ModelArtifact table
  - training.evaluate   — score saved models + generate comparison
                          reports/plots
"""
