"""CLI: run the 11-method comparison (10 baselines + Cross-Balance) on a
chosen dataset with stratified k-fold CV.

    python -m benchmarks.run_benchmark --dataset synthetic:99-1 --folds 10
    python -m benchmarks.run_benchmark --dataset synthetic:all --folds 5
    python -m benchmarks.run_benchmark --dataset real:credit_card_fraud --folds 10
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .datasets import PROJECT_ROOT, SYNTHETIC_RATIOS, load_synthetic
from .methods import build_methods
from .metrics import METRIC_NAMES, compute_metrics
from .real_datasets import REAL_DATASETS, stratified_subsample
from .report import summarize, write_report

SYNTHETIC_RESULTS_DIR = PROJECT_ROOT / "results" / "phase_synthetic"
REAL_RESULTS_DIR = PROJECT_ROOT / "results" / "phase_real"


def run_one_dataset(dataset_key, X, y, folds, seed, methods=None):
    all_methods = build_methods(random_state=seed)
    if methods:
        all_methods = {k: v for k, v in all_methods.items() if k in methods}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = {name: defaultdict(list) for name in all_methods}
    timings = defaultdict(list)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, estimator in all_methods.items():
            try:
                model = clone(estimator)
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - t0

                classes = list(model.classes_)
                pos_col = classes.index(1)
                proba = model.predict_proba(X_test)[:, pos_col]
                pred = model.predict(X_test)

                for metric_name, value in compute_metrics(y_test, pred, proba).items():
                    fold_results[name][metric_name].append(value)
                timings[name].append(fit_time)
            except Exception as exc:  # one method's fold failure shouldn't kill the run
                print(f"  [{dataset_key}] fold {fold_i}: {name} failed: {exc}", file=sys.stderr)
                for metric_name in METRIC_NAMES:
                    fold_results[name][metric_name].append(np.nan)

    summary = summarize(fold_results)
    mean_fit_time = {name: float(np.mean(t)) if t else float("nan") for name, t in timings.items()}
    summary["fit_time_mean_s"] = summary["method"].map(mean_fit_time)
    return summary


def _run_and_report(key, X, y, out_dir, title, folds, seed, methods):
    print(f"=== {title} (folds={folds}) ===")
    print(f"  shape={X.shape}, minority_count={int((y == 1).sum())}")

    summary = run_one_dataset(key, X, y, folds, seed, methods=methods)
    csv_path, html_path = write_report(summary, out_dir, title=f"Cross-Balance Benchmark — {title}")
    print(summary[["rank", "method", "aucpr_mean", "recall_mean", "precision_mean", "mcc_mean"]]
          .to_string(index=False))
    print(f"  -> {csv_path}\n  -> {html_path}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic:99-1",
                         help="synthetic:<99-1|99-2|99-5|all> or real:<name> (see DATA_SOURCES.md)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=20_000,
                         help="synthetic datasets only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="*", default=None,
                         help="subset of method names to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=50_000,
                         help="real datasets only: cap dataset size via stratified subsampling")
    parser.add_argument("--out-root", default=None,
                         help="override results root (default: results/phase_synthetic, results/phase_real). "
                              "Use e.g. results/iteration_1 to avoid overwriting a prior snapshot.")
    args = parser.parse_args(argv)

    synthetic_results_dir = Path(args.out_root) / "synthetic" if args.out_root else SYNTHETIC_RESULTS_DIR
    real_results_dir = Path(args.out_root) / "real" if args.out_root else REAL_RESULTS_DIR

    kind, _, key_arg = args.dataset.partition(":")
    if kind == "synthetic":
        ratio_keys = list(SYNTHETIC_RATIOS) if key_arg == "all" else [key_arg]
        for key in ratio_keys:
            if key not in SYNTHETIC_RATIOS:
                parser.error(f"Unknown synthetic dataset '{key}'. Choose from {list(SYNTHETIC_RATIOS)}")
            X, y = load_synthetic(key, n_samples=args.n_samples, random_state=args.seed)
            _run_and_report(key, X, y, synthetic_results_dir / key, f"synthetic {key}",
                             args.folds, args.seed, args.methods)
    elif kind == "real":
        real_keys = list(REAL_DATASETS) if key_arg == "all" else [key_arg]
        for key in real_keys:
            if key not in REAL_DATASETS:
                parser.error(f"Unknown real dataset '{key}'. Choose from {list(REAL_DATASETS)}")
            try:
                X, y = REAL_DATASETS[key]()
            except FileNotFoundError as exc:
                parser.exit(1, f"{exc}\n")
            X, y = stratified_subsample(X, y, args.max_samples, random_state=args.seed)
            _run_and_report(key, X, y, real_results_dir / key, f"real:{key}",
                             args.folds, args.seed, args.methods)
    else:
        parser.error("--dataset must start with 'synthetic:' or 'real:'")


if __name__ == "__main__":
    main()
