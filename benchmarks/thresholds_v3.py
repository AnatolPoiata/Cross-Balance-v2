"""Internal F1-optimal decision thresholds for every baseline.

For a like-for-like comparison of threshold metrics, every baseline gets the
same rule that ``CrossBalanceClassifier(decision_threshold='f1')`` uses:
inside each outer training fold, a stratified 3-fold split produces
out-of-fold scores, and the threshold maximising F1 on them is applied to
the stored outer-test probabilities of the model fitted on the full
training fold (results/v3/preds). Only training data are used.

    python -m benchmarks.thresholds_v3 --jobs 8
"""

import argparse
import json
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .analyze_v3 import ARTICLE_METHODS
from .run_v3 import N_FOLDS, OUT_DIR, SEEDS, build_method, dataset_names, load_cached

THR_DIR = OUT_DIR / "thresholds_f1"
BASELINES = [m for m in ARTICLE_METHODS if m != "CrossBalance_v3"]


def f1_threshold(y, s):
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def run_job(ds, method, repeat, fold):
    warnings.filterwarnings("ignore")
    out = THR_DIR / f"{ds}__{method}__{repeat}_{fold}.json"
    X, y = load_cached(ds)
    # same outer split as benchmarks.run_v3 (also covers folds whose outer fit failed)
    tr, _ = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=len(SEEDS),
                                         random_state=SEEDS[0]).split(X, y))[repeat * N_FOLDS + fold]
    Xtr, ytr = X[tr], y[tr]
    seed = SEEDS[repeat]
    rec = dict(ds=ds, method=method, repeat=repeat, fold=fold)
    t0 = time.perf_counter()
    try:
        scores = np.zeros(len(ytr))
        for itr, iva in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
            spw = float((ytr[itr] == 0).sum() / max((ytr[itr] == 1).sum(), 1))
            est = build_method(method, spw, seed)
            est.fit(Xtr[itr], ytr[itr])
            scores[iva] = est.predict_proba(Xtr[iva])[:, list(est.classes_).index(1)]
        rec.update(threshold=f1_threshold(ytr, scores), ok=True)
    except Exception as exc:  # e.g. RUSBoost's AdaBoost abort: fall back to 0.5, logged
        rec.update(threshold=0.5, ok=False, error=repr(exc))
    rec["seconds"] = time.perf_counter() - t0
    out.write_text(json.dumps(rec))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    THR_DIR.mkdir(parents=True, exist_ok=True)
    names = dataset_names("all")
    sizes = {ds: len(load_cached(ds)[1]) for ds in names}
    done = {p.stem for p in THR_DIR.glob("*.json") if not p.name.startswith("._")}
    jobs = [(ds, m, r, f) for ds in sorted(names, key=lambda d: -sizes[d]) for m in BASELINES
            for r in range(len(SEEDS)) for f in range(5) if f"{ds}__{m}__{r}_{f}" not in done]
    print(f"{len(jobs)} threshold jobs", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_job)(*j) for j in jobs)
    print(f"finished in {time.time() - t0:.0f}s; fallbacks: {sum(not r['ok'] for r in res)}", flush=True)


if __name__ == "__main__":
    main()
