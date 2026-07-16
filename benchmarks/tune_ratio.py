"""Independently tune CrossBalanceClassifier's majority:minority chunk ratio.

Every prior iteration (0-7) used ratio=2.0 (each chunk pairs the full
minority class with 2x as many majority samples) without ever testing
whether that value is actually best -- it was carried over unchanged from
the original v1 prototype. This script sweeps ratio in {1, 2, 3, 4, 5}
against the same held-out synthetic tuning suite used for threshold tuning
(benchmarks/tune_thresholds.py) -- disjoint datasets and random seeds from
the 11+4-dataset benchmark -- so the winning value is validated
independently before being retested on the real benchmark, the same
methodology that resolved iteration 6's regression in iteration 7.

    python -m benchmarks.tune_ratio
"""

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .datasets import make_synthetic_imbalanced
from .tune_thresholds import TUNE_FOLDS, TUNE_N_CHUNKS, TUNE_SEED, TUNING_DATASETS
from cross_balance import CrossBalanceClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RATIO_CANDIDATES = [1.0, 2.0, 3.0, 4.0, 5.0]

# Iteration 7's validated thresholds/max_chunk_size are held fixed here --
# this script isolates the effect of `ratio` alone, not a joint retuning.
TUNED_THRESHOLDS = (2.9, 12.5, 19.3, 238.6)


def score_ratio(ratio, datasets):
    scores = []
    for X, y in datasets:
        clf = CrossBalanceClassifier(
            base_estimator="auto", auto_density_thresholds=TUNED_THRESHOLDS,
            ensemble_type="boosting", ratio=ratio, n_chunks=TUNE_N_CHUNKS,
            weighting="oof", max_chunk_size=3000, random_state=TUNE_SEED,
        )
        skf = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=TUNE_SEED)
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, list(clf.classes_).index(1)]
            scores.append(average_precision_score(y[test_idx], proba))
    return float(np.mean(scores))


def main():
    print("=== Building tuning suite (same as tune_thresholds.py, disjoint from the benchmark) ===")
    datasets = []
    for cfg in TUNING_DATASETS:
        X, y = make_synthetic_imbalanced(**cfg)
        datasets.append((X, y))
        print(f"  n={cfg['n_samples']} feat={cfg['n_features']} ratio={cfg['ratio']} "
              f"seed={cfg['random_state']} -> minority={int(y.sum())}")

    print(f"\n=== Scoring ratio in {RATIO_CANDIDATES} on the tuning suite ===")
    results = []
    for ratio in RATIO_CANDIDATES:
        t0 = time.perf_counter()
        score = score_ratio(ratio, datasets)
        dt = time.perf_counter() - t0
        tag = " (all prior iterations' default)" if ratio == 2.0 else ""
        print(f"  ratio={ratio:.1f} -> mean AP={score:.4f} ({dt:.1f}s){tag}")
        results.append((ratio, score))

    results.sort(key=lambda r: -r[1])
    print("\n=== Ranked ===")
    for ratio, score in results:
        print(f"  ratio={ratio:.1f} -> {score:.4f}")

    best_ratio, best_score = results[0]
    print(f"\nBest: ratio={best_ratio} (mean AP={best_score:.4f} on tuning suite)")

    out_path = PROJECT_ROOT / "results" / "ratio_tuning.txt"
    with open(out_path, "w") as f:
        f.write("Independent ratio tuning for CrossBalanceClassifier (majority:minority per chunk)\n")
        f.write(f"Tuning suite (disjoint from the benchmark datasets): {TUNING_DATASETS}\n")
        f.write(f"Thresholds held fixed at iteration 7's validated value: {TUNED_THRESHOLDS}\n\n")
        for ratio, score in results:
            f.write(f"{ratio}\t{score:.4f}\n")
        f.write(f"\nBest: {best_ratio}\t{best_score:.4f}\n")
    print(f"-> {out_path}")
    return best_ratio


if __name__ == "__main__":
    main()
