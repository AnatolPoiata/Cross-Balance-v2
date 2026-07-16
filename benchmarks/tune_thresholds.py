"""Independently tune CrossBalanceClassifier's 5-tier auto_density_thresholds.

Iteration 6 (see results/iteration_6) tested the 5-tier auto ladder
(OMP/LR/tree/RF/GB) with thresholds (5.0, 10.0, 30.0, 100.0) chosen once, by
inspection, on the same 11 datasets used to report results -- a methodological
flaw flagged in the article's Limitations section. This script resolves that
by tuning thresholds on a *separate* synthetic tuning suite (different
random seeds, different sizes/dimensionality/ratios than any of the 11
benchmark datasets), then the tuned thresholds are evaluated on the full
benchmark completely independently of the tuning process.

    python -m benchmarks.tune_thresholds
"""

import random
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from .datasets import make_synthetic_imbalanced
from cross_balance import CrossBalanceClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tuning suite: deliberately different sizes/dimensionality/ratios and a
# disjoint random_state range (1001+) from the main benchmark (42), so
# nothing here can leak into the 11-dataset test set.
TUNING_DATASETS = [
    dict(n_samples=8_000, n_features=15, n_informative=10, n_redundant=2, ratio=0.02, random_state=1001),
    dict(n_samples=30_000, n_features=25, n_informative=18, n_redundant=3, ratio=0.05, random_state=1002),
    dict(n_samples=60_000, n_features=10, n_informative=7, n_redundant=1, ratio=0.01, random_state=1003),
    dict(n_samples=15_000, n_features=40, n_informative=25, n_redundant=5, ratio=0.08, random_state=1004),
]

N_CANDIDATES = 20
TUNE_N_CHUNKS = 20  # reduced from the validated 50 purely for tuning-phase speed
TUNE_FOLDS = 3
TUNE_SEED = 999  # disjoint from both the benchmark's 42 and the tuning data's 1001-1004


def sample_thresholds(rng):
    t_omp = rng.uniform(1.0, 8.0)
    t_lr = rng.uniform(t_omp + 2.0, 20.0)
    t_tree = rng.uniform(t_lr + 5.0, 60.0)
    t_rf = rng.uniform(t_tree + 10.0, 300.0)
    return (round(t_omp, 1), round(t_lr, 1), round(t_tree, 1), round(t_rf, 1))


def score_thresholds(thresholds, datasets):
    scores = []
    for X, y in datasets:
        clf = CrossBalanceClassifier(
            base_estimator="auto", auto_density_thresholds=thresholds,
            ensemble_type="boosting", ratio=2.0, n_chunks=TUNE_N_CHUNKS,
            weighting="oof", max_chunk_size=3000, random_state=TUNE_SEED,
        )
        skf = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=TUNE_SEED)
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, list(clf.classes_).index(1)]
            scores.append(average_precision_score(y[test_idx], proba))
    return float(np.mean(scores))


def main():
    print("=== Building tuning suite (separate from the 11 benchmark datasets) ===")
    datasets = []
    for cfg in TUNING_DATASETS:
        X, y = make_synthetic_imbalanced(**cfg)
        datasets.append((X, y))
        print(f"  n={cfg['n_samples']} feat={cfg['n_features']} ratio={cfg['ratio']} "
              f"seed={cfg['random_state']} -> minority={int(y.sum())}")

    rng = random.Random(TUNE_SEED)
    candidates = [(5.0, 10.0, 30.0, 100.0)]  # include iteration 6's original as a reference point
    candidates += [sample_thresholds(rng) for _ in range(N_CANDIDATES)]

    print(f"\n=== Scoring {len(candidates)} threshold candidates on the tuning suite ===")
    results = []
    for i, thresholds in enumerate(candidates):
        t0 = time.perf_counter()
        score = score_thresholds(thresholds, datasets)
        dt = time.perf_counter() - t0
        tag = " (iteration-6 original)" if i == 0 else ""
        print(f"  [{i:>2}] thresholds={thresholds} -> mean AP={score:.4f} ({dt:.1f}s){tag}")
        results.append((thresholds, score))

    results.sort(key=lambda r: -r[1])
    print("\n=== Top 5 ===")
    for thresholds, score in results[:5]:
        print(f"  {thresholds} -> {score:.4f}")

    best_thresholds, best_score = results[0]
    print(f"\nBest: {best_thresholds} (mean AP={best_score:.4f} on tuning suite)")

    out_path = PROJECT_ROOT / "results" / "threshold_tuning.txt"
    with open(out_path, "w") as f:
        f.write("Independent threshold tuning for CrossBalanceClassifier auto_density_thresholds\n")
        f.write(f"Tuning suite (disjoint from the 11 benchmark datasets): {TUNING_DATASETS}\n\n")
        for thresholds, score in results:
            f.write(f"{thresholds}\t{score:.4f}\n")
        f.write(f"\nBest: {best_thresholds}\t{best_score:.4f}\n")
    print(f"-> {out_path}")
    return best_thresholds


if __name__ == "__main__":
    main()
