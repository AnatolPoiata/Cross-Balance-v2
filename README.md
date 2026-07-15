# Cross-Balance

Cross-Balance is an ensemble method for imbalanced binary classification. It
partitions the majority class into balanced-ratio chunks, pairs each chunk
with the full minority class, trains one base estimator per chunk, and
aggregates predictions via out-of-fold performance-weighted averaging.

This is v2 of the project — a clean, packaged, tested rebuild of the
prototype in the sibling `Cross-Balance/` folder (`cbc_v1.py` → `cbc_v2.py`),
which already showed Cross-Balance competitive with SMOTE/EasyEnsemble/BRF on
a synthetic 99:1 dataset and the Kaggle Credit Card Fraud dataset.

## Project roadmap

- **Phase A** (this repo, in progress): package the algorithm as a library,
  build a benchmarking harness, generate synthetic imbalanced datasets
  (99:1, 99:2, 99:5), and compare Cross-Balance against 10 established
  imbalance-handling methods.
- **Phase B**: extend the same comparison to 7 real-world datasets
  (Communities & Crime, Credit Card Fraud, NIH Chest X-ray sample, Bioassay,
  Synthetic Financial Fraud, Fraud Detection Transactions, Credit Card Fraud
  2023). Requires manually downloading each dataset into `data/raw/<name>/`.
- **Phase C**: iterate on the algorithm for 5 rounds guided by benchmark
  results, then write up findings as a research-style article in `article/`.

## Install

```bash
pip install -e ".[boost,plot,dev]"
```

## Usage

```python
from cross_balance import CrossBalanceClassifier
from sklearn.linear_model import LogisticRegression

clf = CrossBalanceClassifier(
    base_estimator=LogisticRegression(max_iter=1000),
    ensemble_type="bagging",   # or "boosting"
    ratio=2.0,
    n_chunks=10,
    weighting="oof",           # or "uniform"
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

## Benchmarks

```bash
python -m benchmarks.run_benchmark --dataset synthetic --ratio 0.01 --folds 10
```

Results land in `results/phase_synthetic/<ratio>/` as `results_summary.csv`
and `report.html`.

## Layout

```
src/cross_balance/   the library
benchmarks/           dataset loaders, method registry, CLI, reporting
data/raw/             manually-downloaded real datasets (gitignored)
data/processed/        cached/generated datasets (gitignored)
results/               benchmark outputs
tests/                 pytest suite
article/               Phase C research article draft
references/            literature corpus (gitignored from git, kept locally)
```
