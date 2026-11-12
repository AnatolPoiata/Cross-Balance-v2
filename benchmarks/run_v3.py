"""v3 evaluation: Cross-Balance v0.2 vs. strong baselines, repeated CV.

Protocol (fixed before any hold-out dataset was loaded):
  * repeated stratified 5-fold CV, 3 repeats (seeds 1, 2, 3) -> 15 paired
    test folds per (dataset, method);
  * development datasets (3 synthetic + 12 public sets that guided v1-v3)
    and hold-out datasets (ULB credit-card fraud + the 16-set Park & Im
    collection, first loaded after the v3 design freeze) are reported
    separately;
  * datasets above 50,000 rows are stratified-subsampled as in the earlier
    benchmark, except ``creditcard_ulb``, which is run at full size and
    native prevalence (it carries the transaction amounts used by the
    cost analysis);
  * every method sees identical folds; scaling and resampling happen inside
    the training fold only (imblearn Pipelines).

Per-fold test scores and labels are saved so that every metric (AP,
ROC-AUC, Brier, Precision@k, cost curves, statistical tests) is computed
from the same stored predictions by ``benchmarks.analyze_v3``.

    python -m benchmarks.run_v3 --group all --jobs 8
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from .datasets import PROJECT_ROOT, SYNTHETIC_RATIOS, load_synthetic
from .real_datasets import DEVELOPMENT_DATASETS, HOLDOUT_DATASETS, REAL_DATASETS, stratified_subsample

OUT_DIR = PROJECT_ROOT / "results" / "v3"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "v3_cache"
FULL_SIZE = {"creditcard_ulb"}
SEEDS = (1, 2, 3)
N_FOLDS = 5

MAIN_METHODS = ["LR_balanced", "SMOTE_LR", "RF_balanced", "BalancedRF", "EasyEnsemble", "RUSBoost",
                "XGBoost_spw", "LightGBM_spw", "LightGBM", "SPE", "CrossBalance_v2", "CrossBalance_v3"]
ABLATIONS = ["CB3_bagging", "CB3_boosting_v2", "CB3_hgb_only", "CB3_auto_ladder", "CB3_ratio2",
             "CB3_oof_weights", "CB3_no_cap", "CB3_f1_threshold"]


def build_method(name, spw, seed):
    from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                   RUSBoostClassifier)
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from cross_balance import LEGACY_V2_PARAMS, CrossBalanceClassifier

    from .spe import SelfPacedEnsembleClassifier

    cb = dict(random_state=seed)
    m = {
        "LR_balanced": lambda: Pipeline([("sc", StandardScaler()),
                                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "SMOTE_LR": lambda: Pipeline([("sc", StandardScaler()), ("sm", SMOTE(random_state=seed)),
                                      ("lr", LogisticRegression(max_iter=2000))]),
        "RF_balanced": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                      min_samples_leaf=2, n_jobs=1, random_state=seed),
        "BalancedRF": lambda: BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all",
                                                             replacement=True, bootstrap=False,
                                                             n_jobs=1, random_state=seed),
        "EasyEnsemble": lambda: EasyEnsembleClassifier(n_estimators=10, n_jobs=1, random_state=seed),
        "RUSBoost": lambda: RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                                               n_estimators=100, random_state=seed),
        "XGBoost_spw": lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                             scale_pos_weight=spw, n_jobs=1, random_state=seed),
        "LightGBM_spw": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, scale_pos_weight=spw,
                                               verbose=-1, n_jobs=1, random_state=seed),
        "LightGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, n_jobs=1,
                                           random_state=seed),
        "SPE": lambda: SelfPacedEnsembleClassifier(n_estimators=50, random_state=seed),
        "CrossBalance_v2": lambda: CrossBalanceClassifier(**LEGACY_V2_PARAMS, **cb),
        "CrossBalance_v3": lambda: CrossBalanceClassifier(**cb),
        "CB3_bagging": lambda: CrossBalanceClassifier(ensemble_type="bagging", **cb),
        "CB3_boosting_v2": lambda: CrossBalanceClassifier(ensemble_type="boosting", **cb),
        "CB3_hgb_only": lambda: CrossBalanceClassifier(base_estimator="hgb", **cb),
        "CB3_auto_ladder": lambda: CrossBalanceClassifier(base_estimator="auto", **cb),
        "CB3_ratio2": lambda: CrossBalanceClassifier(ratio=2.0, **cb),
        "CB3_oof_weights": lambda: CrossBalanceClassifier(weighting="oof", **cb),
        "CB3_no_cap": lambda: CrossBalanceClassifier(max_chunk_size=None, **cb),
        "CB3_f1_threshold": lambda: CrossBalanceClassifier(decision_threshold="f1", **cb),
    }
    return m[name]()


def dataset_names(group):
    dev = [f"syn_{k}" for k in SYNTHETIC_RATIOS] + list(DEVELOPMENT_DATASETS)
    return {"dev": dev, "holdout": list(HOLDOUT_DATASETS), "all": dev + list(HOLDOUT_DATASETS)}[group]


def load_cached(name):
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if name.startswith("syn_"):
            X, y = load_synthetic(name[4:])
        else:
            X, y = REAL_DATASETS[name]()
            if name not in FULL_SIZE:
                X, y = stratified_subsample(X, y, 50_000, random_state=42)
        np.savez_compressed(path, X=X.astype(np.float64), y=y.astype(np.int8))
    d = np.load(path)
    return d["X"], d["y"].astype(int)


def run_job(ds, method, repeat, fold, out_dir):
    warnings.filterwarnings("ignore")
    X, y = load_cached(ds)
    splits = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=len(SEEDS),
                                          random_state=SEEDS[0]).split(X, y))
    tr, te = splits[repeat * N_FOLDS + fold]
    spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    seed = SEEDS[repeat]
    rec = dict(ds=ds, method=method, repeat=repeat, fold=fold)
    try:
        est = build_method(method, spw, seed)
        t0 = time.perf_counter()
        est.fit(X[tr], y[tr])
        rec["fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = est.predict_proba(X[te])[:, list(est.classes_).index(1)]
        rec["pred_s"] = time.perf_counter() - t0
        pred = est.predict(X[te]).astype(int)
        np.savez_compressed(out_dir / "preds" / f"{ds}__{method}__{repeat}_{fold}.npz",
                            idx=te.astype(np.int32), proba=proba.astype(np.float32), pred=pred.astype(np.int8))
        rec["ok"] = True
    except Exception as exc:  # logged, never silently dropped
        rec.update(ok=False, error=repr(exc))
    (out_dir / "preds" / f"{ds}__{method}__{repeat}_{fold}.json").write_text(json.dumps(rec))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="all", choices=["dev", "holdout", "all"])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--ablations", action="store_true", help="also run the CB3_* ablation variants")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = OUT_DIR
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    methods = args.methods or (MAIN_METHODS + (ABLATIONS if args.ablations else []))
    names = args.datasets or dataset_names(args.group)
    sizes = {}
    for ds in names:  # build caches serially (loaders are memory-hungry)
        X, y = load_cached(ds)
        sizes[ds] = len(y)
        print(f"  {ds:30s} {X.shape} pos={int(y.sum())} ({100 * y.mean():.2f}%)", flush=True)

    done = set()
    for p in (out_dir / "preds").glob("*.json"):
        if p.name.startswith("._"):
            continue
        r = json.loads(p.read_text())
        if r.get("ok"):
            done.add((r["ds"], r["method"], r["repeat"], r["fold"]))
    jobs = [(ds, m, r, f) for ds in sorted(names, key=lambda d: -sizes[d]) for m in methods
            for r in range(len(SEEDS)) for f in range(N_FOLDS) if (ds, m, r, f) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_job)(*j, out_dir) for j in jobs)
    failed = [r for r in results if not r["ok"]]
    print(f"finished in {time.time() - t0:.0f}s; failures: {len(failed)}", flush=True)
    for r in failed:
        print("  FAILED", r, file=sys.stderr)


if __name__ == "__main__":
    main()
