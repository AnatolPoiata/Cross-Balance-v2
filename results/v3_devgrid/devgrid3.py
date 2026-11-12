import sys, time, json, numpy as np, warnings, itertools; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/WORK/__Articles/Cross-Balance-v2"); sys.path.insert(0, ".")
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from proto import CBProto

DEV = ["syn_99-1","syn_98-2","syn_95-5","credit_card_fraud","synthetic_financial_fraud","ailerons",
       "communities_crime","bank_marketing","default_credit_card_clients","mammographic"]

def references(spw):
    from cross_balance import CrossBalanceClassifier
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    return {
      "CB_v2": CrossBalanceClassifier(random_state=0),
      "BRF": BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
      "RF_bal": RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=1, random_state=0),
      "LGBM_spw": LGBMClassifier(n_estimators=400, learning_rate=0.05, scale_pos_weight=spw, verbose=-1, n_jobs=1, random_state=0),
      "XGB_spw": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, scale_pos_weight=spw, n_jobs=1, random_state=0),
      "SPE": CBProto(base="dt_full", ratio=1.0, cap=None, mode="selfpaced", n_chunks=50, random_state=0),
    }

def run(ds, fold, cfg_name, cfg):
    d = np.load(f"cache/{ds}.npz"); X, y = d["X"].astype(float), d["y"]
    tr, te = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))[fold]
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    est = references(spw)[cfg_name] if cfg is None else CBProto(random_state=0, **cfg)
    t = time.time(); est.fit(X[tr], y[tr]); dt = time.time() - t
    p = est.predict_proba(X[te])[:, 1]
    return dict(ds=ds, fold=fold, cfg=cfg_name, ap=average_precision_score(y[te], p), t=dt)

if __name__ == "__main__":
    grid = {
      "sp_mix_r3": dict(mode="selfpaced", base="mix", ratio=3.0),
      "sp_mix_r4": dict(mode="selfpaced", base="mix", ratio=4.0),
    }
    jobs = [(ds, f, n, c) for ds in DEV for f in range(3) for n, c in grid.items()]
    out = Parallel(n_jobs=8, verbose=0)(delayed(run)(*j) for j in jobs)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "dev1.json", "w"))
