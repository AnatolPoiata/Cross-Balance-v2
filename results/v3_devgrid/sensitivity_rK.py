"""Sensitivity of the frozen configuration to r > 4 and K > 50, on the same 10
selection datasets and protocol as the configuration search (3-fold CV, seed 0).
Uses the library itself (CrossBalanceClassifier), not the prototype.

    ./.venv/bin/python results/v3_devgrid/sensitivity_rK.py
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.run_v3 import load_cached  # noqa: E402
from cross_balance import CrossBalanceClassifier  # noqa: E402

SEL = ["syn_99-1", "syn_98-2", "syn_95-5", "credit_card_fraud", "synthetic_financial_fraud", "ailerons",
       "communities_crime", "bank_marketing", "default_credit_card_clients", "mammographic"]
GRID = {"r4_K50 (default)": dict(ratio=4.0, n_chunks=50), "r5_K50": dict(ratio=5.0, n_chunks=50),
        "r6_K50": dict(ratio=6.0, n_chunks=50), "r8_K50": dict(ratio=8.0, n_chunks=50),
        "r4_K75": dict(ratio=4.0, n_chunks=75), "r4_K100": dict(ratio=4.0, n_chunks=100)}


def job(ds, fold, name):
    warnings.filterwarnings("ignore")
    X, y = load_cached(ds)
    tr, te = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))[fold]
    t = time.time()
    m = CrossBalanceClassifier(random_state=0, **GRID[name]).fit(X[tr], y[tr])
    return dict(ds=ds, fold=fold, cfg=name, ap=average_precision_score(y[te], m.predict_proba(X[te])[:, 1]),
                t=time.time() - t)


if __name__ == "__main__":
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res = Parallel(n_jobs=nj)(delayed(job)(d, f, n) for d in SEL for f in range(3) for n in GRID)
    out = Path(__file__).with_name("sensitivity_rK.json")
    out.write_text(json.dumps(res, indent=1))
    import pandas as pd
    df = pd.DataFrame(res)
    piv = df.groupby(["ds", "cfg"]).ap.mean().unstack()
    summ = pd.DataFrame({"mean_ap": piv.mean(), "avg_rank": piv.rank(axis=1, ascending=False).mean(),
                         "fit_s": df.groupby("cfg").t.mean()}).sort_values("avg_rank")
    print(summ.round(4).to_string())
    summ.to_csv(Path(__file__).with_name("sensitivity_rK.csv"))
