"""Prototype Cross-Balance v3 with switchable components for the dev grid."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

def make_base(name, rs):
    if name == "rf":  return RandomForestClassifier(n_estimators=20, max_depth=8, max_samples=0.3, n_jobs=1, random_state=rs)
    if name == "rf_deep": return RandomForestClassifier(n_estimators=20, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=rs)
    if name == "et":  return ExtraTreesClassifier(n_estimators=20, min_samples_leaf=2, n_jobs=1, random_state=rs)
    if name == "hgb": return HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, max_leaf_nodes=15, early_stopping=False, random_state=rs)
    if name == "dt":  return DecisionTreeClassifier(max_depth=6, random_state=rs)
    if name == "dt_full": return DecisionTreeClassifier(random_state=rs)
    raise ValueError(name)

def spe_bins(hard, n_target, alpha, rng, k_bins=5):
    """Self-paced hardness-harmonised sampling (Liu et al., 2020)."""
    if hard.max() == hard.min():
        return rng.choice(len(hard), n_target, replace=len(hard) < n_target)
    pops, edges = np.histogram(hard, bins=k_bins)
    idx_bins, contrib = [], np.zeros(k_bins)
    for b in range(k_bins):
        m = (hard >= edges[b]) & (hard < edges[b+1]) if b < k_bins-1 else (hard >= edges[b]) & (hard <= edges[b+1])
        idx_bins.append(np.where(m)[0])
        if pops[b] > 0: contrib[b] = hard[m].mean()
    with np.errstate(divide="ignore"):
        w = 1.0 / (contrib + alpha)
    w[(pops == 0) | ~np.isfinite(w)] = 0
    n_b = np.floor(n_target * w / w.sum()).astype(int)
    order = np.argsort(-pops); i = 0
    while n_b.sum() < n_target:
        b = order[i % k_bins]
        if pops[b] > 0: n_b[b] += 1
        i += 1
    out = [rng.choice(idx_bins[b], n_b[b], replace=n_b[b] > pops[b]) for b in range(k_bins) if n_b[b] > 0]
    return np.concatenate(out)

class CBProto(ClassifierMixin, BaseEstimator):
    def __init__(self, base="rf", ratio=2.0, n_chunks=50, mode="selfpaced", cap=3000,
                 weighting="uniform", min_boot=False, prior_correct=False, threshold="half",
                 random_state=None):
        self.base=base; self.ratio=ratio; self.n_chunks=n_chunks; self.mode=mode; self.cap=cap
        self.weighting=weighting; self.min_boot=min_boot; self.prior_correct=prior_correct
        self.threshold=threshold; self.random_state=random_state

    def _fit_core(self, X, y):
        rng = check_random_state(self.random_state)
        Xmin, Xmaj = X[y == 1], X[y == 0]
        n_min, n_maj = len(Xmin), len(Xmaj)
        n_maj_c = min(max(1, int(round(self.ratio * n_min))), n_maj)
        n_min_c = n_min
        if self.cap and n_min_c + n_maj_c > self.cap:
            s = self.cap / (n_min_c + n_maj_c)
            n_min_c = max(1, int(round(n_min * s))); n_maj_c = max(1, self.cap - n_min_c)
        self.beta_ = n_maj_c / n_maj          # majority sampling rate (Dal Pozzolo prior correction)
        self.gamma_ = n_min_c / n_min         # minority sampling rate
        F = np.zeros(n_maj)                   # running ensemble score of majority samples
        w_boost = np.ones(n_maj)
        ests, ws = [], []
        for i in range(self.n_chunks):
            seed = rng.randint(2**31 - 1)
            r = np.random.RandomState(seed)
            if self.mode == "bagging":
                maj_idx = r.choice(n_maj, n_maj_c, replace=False)
            elif self.mode == "boost_v2":
                maj_idx = r.choice(n_maj, n_maj_c, replace=True, p=w_boost / w_boost.sum())
            elif self.mode == "selfpaced":
                alpha = np.tan(np.pi * 0.5 * i / max(self.n_chunks - 1, 1))
                maj_idx = spe_bins(F, n_maj_c, alpha, r)
            elif self.mode == "boost_ens":   # soft, bounded hardness on the ensemble-so-far
                p = 0.1 + F if i > 0 else np.ones(n_maj)
                maj_idx = r.choice(n_maj, n_maj_c, replace=False, p=p / p.sum())
            if n_min_c < n_min or self.min_boot:
                min_idx = r.choice(n_min, n_min_c, replace=self.min_boot)
            else:
                min_idx = np.arange(n_min)
            Xc = np.vstack([Xmin[min_idx], Xmaj[maj_idx]])
            yc = np.r_[np.ones(len(min_idx)), np.zeros(len(maj_idx))].astype(int)
            bname = (["hgb", "et"][i % 2] if self.base == "mix" else self.base)
            est = make_base(bname, seed).fit(Xc, yc)
            if self.weighting == "oob":   # held-out majority not in chunk + minority OOB if bootstrapped
                oob = np.setdiff1d(np.arange(n_maj), maj_idx)
                oob = r.choice(oob, min(len(oob), 5 * n_min), replace=False) if len(oob) else oob
                Xv = np.vstack([Xmin, Xmaj[oob]]); yv = np.r_[np.ones(n_min), np.zeros(len(oob))]
                w = average_precision_score(yv, est.predict_proba(Xv)[:, 1])
            else:
                w = 1.0
            ests.append(est); ws.append(w)
            if self.mode in ("selfpaced", "boost_ens") or self.mode == "boost_v2":
                pm = est.predict_proba(Xmaj)[:, 1]
                F = (F * i + pm) / (i + 1)
                if self.mode == "boost_v2":
                    w_boost *= 1.0 + (pm > 0.5); w_boost /= w_boost.sum()
        ws = np.asarray(ws); self.estimators_ = ests; self.weights_ = ws / ws.sum()
        return self

    def _raw(self, X):
        return sum(w * e.predict_proba(X)[:, 1] for e, w in zip(self.estimators_, self.weights_))

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        if self.threshold == "half":
            self.thr_ = 0.5
        else:   # internal 3-fold: choose F1-optimal threshold on held-out (calibrated) scores
            scores = np.zeros(len(y))
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=self.random_state).split(X, y):
                m = clone(self).set_params(threshold="half")._fit_core(X[tr], y[tr])
                scores[va] = m._post(m._raw(X[va]))
            from sklearn.metrics import precision_recall_curve
            P, R, T = precision_recall_curve(y, scores)
            f1 = 2 * P * R / np.maximum(P + R, 1e-12)
            self.thr_ = float(T[np.argmax(f1[:-1])])
        return self._fit_core(X, y)

    def _post(self, s):
        if not self.prior_correct: return s
        # undo the sampling bias: odds scale by gamma/beta (Dal Pozzolo et al., 2015, generalised)
        k = self.beta_ / self.gamma_
        return k * s / (k * s - s + 1)

    def predict_proba(self, X):
        p = self._post(self._raw(X)); return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.thr_).astype(int)
