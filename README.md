# Cross-Balance

Cross-Balance is an ensemble method for imbalanced binary classification. It
partitions the majority class into balanced-ratio chunks, pairs each chunk
with the full minority class, trains one base estimator per chunk, and
aggregates predictions via out-of-fold performance-weighted averaging.

This is v2 of the project — a clean, packaged, tested rebuild of the
prototype in the sibling `Cross-Balance/` folder (`cbc_v1.py` → `cbc_v2.py`).

## Project roadmap

- **Phase A** (done): packaged the algorithm as a library, built a
  benchmarking harness, generated synthetic imbalanced datasets (99:1,
  98:2, 95:5, 50,000 rows each), and compared Cross-Balance against 10
  established imbalance-handling methods. See `results/phase_synthetic/`.
- **Phase B** (done): extended the comparison to 8 real-world datasets —
  Communities & Crime, Credit Card Fraud (kartik2112), Credit Card Fraud
  2023, PaySim (synthetic financial fraud), Fraud Detection Transactions,
  Ailerons, and KEEL's segment0 / shuttle_2_vs_5. `bioassay` and
  `nih_chest_xray_sample` were excluded (see `DATA_SOURCES.md`). See
  `results/phase_real/`.
- **Phase C** (done — 6 rounds): iterated on the algorithm guided by
  benchmark results. See `results/iteration_1/` through `iteration_6/`,
  each commit documents what changed and why. Validated config (the
  library defaults): `base_estimator='auto'` (2-tier LR/tree ladder),
  `ensemble_type='boosting'`, `n_chunks=50`. Cross-Balance finishes rank 1
  on 5 of 11 benchmark datasets and statistically ties the strongest
  baseline (Balanced Random Forest) on a 6th — see commit `a3dc814` for
  the full tally. Iteration 6 extended `auto` to a 5-tier ladder (adding
  OMP/RandomForest/GradientBoosting) and is a documented **negative
  result** — it regressed 6 of 11 datasets and is not the shipped default,
  though it remains available via an explicit `auto_density_thresholds`
  override.
- **Article**: a full draft is in `article/draft.md`, covering the method,
  all 6 iterations (including iteration 6's negative result), and honest
  limitations (see `REFERENCES.md` for source material).

## Install

```bash
pip install -e ".[boost,plot,bench,dev]"
```

## Usage

```python
from cross_balance import CrossBalanceClassifier

# Defaults reflect the best config validated across Phase C's 6 benchmark rounds
# (iteration 6 tested a richer estimator ladder and regressed -- not adopted).
clf = CrossBalanceClassifier(random_state=42)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

## Benchmarks

```bash
python -m benchmarks.run_benchmark --dataset synthetic:99-1 --folds 10
python -m benchmarks.run_benchmark --dataset synthetic:all --folds 10
python -m benchmarks.run_benchmark --dataset real:credit_card_fraud --folds 10
python -m benchmarks.run_benchmark --dataset real:all --max-samples 50000
```

Results land in `results/phase_synthetic/<ratio>/` or `results/phase_real/<name>/`
as `results_summary.csv` and `report.html`. Pass `--out-root results/iteration_N`
to avoid overwriting a prior snapshot when iterating on the method.

## Layout

```
src/cross_balance/     the library
benchmarks/             dataset loaders, method registry, CLI, reporting
data/raw/                manually-downloaded real datasets (gitignored) -- see DATA_SOURCES.md
data/processed/          cached/generated synthetic datasets (gitignored)
results/                 benchmark outputs (phase_synthetic, phase_real, iteration_1..6)
tests/                   pytest suite
article/                 Phase C research article draft
references/              literature corpus (gitignored from git, kept locally)
DATA_SOURCES.md          per-dataset download links, schemas, known caveats
REFERENCES.md            external reading list for the article
```
