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
- **Phase C** (done — 7 rounds): iterated on the algorithm guided by
  benchmark results. See `results/iteration_1/` through `iteration_7/`,
  each commit documents what changed and why. Validated config (the
  library defaults): `base_estimator='auto'` (5-tier OMP/LR/tree/RF/GB
  ladder with independently-tuned thresholds), `ensemble_type='boosting'`,
  `n_chunks=50`, `max_chunk_size=3000`. Cross-Balance finishes rank 1 on 6
  of 11 benchmark datasets, including beating the strongest baseline
  (Balanced Random Forest) outright on every synthetic imbalance ratio
  tested — see commit `86a6c18` for the full tally. Iteration 6 first
  extended `auto` to a 5-tier ladder using thresholds chosen by inspection
  and regressed 6 of 11 datasets (a documented **negative result**);
  iteration 7 re-tuned those thresholds against a held-out synthetic suite
  disjoint from the benchmark (`benchmarks/tune_thresholds.py`) and
  resolved most, not all, of that regression — the tuned thresholds are
  now the shipped default.
- **Extended validation** (done): re-tested the shipped iteration-7 config,
  unchanged, against three checks beyond the original 11-dataset benchmark
  — (1) full, non-subsampled reruns of the three largest fraud datasets
  (see `results/iteration_7_fullsize/`): Credit Card Fraud's win gets
  *stronger* at full scale (1.85M rows), PaySim's *reverses* (6.36M rows —
  Balanced Random Forest has no `max_chunk_size`-equivalent cap, so it
  benefits more from the extra data); (2) four new UCI datasets —
  `default_credit_card_clients`, `mammographic`, `mushrooms`,
  `bank_marketing` — competitive on three, a clear miss on `mammographic`
  (traced to its 4-feature count inflating the density heuristic on a
  dataset too small in absolute terms to support the resulting
  high-capacity tier); (3) an independent sweep of `ratio` in {1..5}
  (`benchmarks/tune_ratio.py`, `results/ratio_tuning.txt`) confirms
  `ratio=2.0` — used unchanged and untested across all 7 iterations — was
  already the best choice.
- **Article**: a full draft is in `article/draft.md`, covering the method,
  all 7 iterations (including iteration 6's negative result and iteration
  7's resolution), the extended validation above, and honest limitations
  (see `REFERENCES.md` for source material).

## Install

```bash
pip install -e ".[boost,plot,bench,dev]"
```

## Usage

```python
from cross_balance import CrossBalanceClassifier

# Defaults reflect the best config validated across Phase C's 7 benchmark rounds,
# including iteration 7's independently-tuned auto_density_thresholds.
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
python -m benchmarks.run_benchmark --dataset real:mammographic --folds 10  # new UCI datasets

# Full-size (not subsampled) -- slow; excluding BorderlineSMOTE/SMOTETomek is
# recommended above a few million rows, see DATA_SOURCES.md and article §5.5:
python -m benchmarks.run_benchmark --dataset real:synthetic_financial_fraud --max-samples 999999999 \
  --methods Baseline ClassWeighted RandomUnderSampler RandomOverSampler SMOTE ADASYN BalancedRandomForest EasyEnsemble CrossBalance
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
results/                 benchmark outputs (phase_synthetic, phase_real, iteration_1..7,
                          iteration_7_fullsize, threshold_tuning.txt, ratio_tuning.txt)
tests/                   pytest suite
article/                 Phase C research article draft
references/              literature corpus (gitignored from git, kept locally)
DATA_SOURCES.md          per-dataset download links, schemas, known caveats
REFERENCES.md            external reading list for the article
```
