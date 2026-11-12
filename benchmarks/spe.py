"""Self-Paced Ensemble (Liu et al., ICDE 2020) as a comparison baseline.

Port of the reference implementation in ``imbalanced-ensemble`` 0.2.x
(``SelfPacedEnsembleClassifier`` + ``SelfPacedUnderSampler``, defaults
k_bins=5, soft_resample_flag=False, replacement=True), which cannot be
installed next to scikit-learn >= 1.7. Base estimator, as in the
reference: an unpruned ``DecisionTreeClassifier``.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def self_paced_sample(hardness, n_target, alpha, rng, k_bins=5, replace=True):
    """Hardness-harmonised under-sampling of one class (SPE, Alg. 1)."""
    if hardness.max() == hardness.min() or n_target >= len(hardness):
        return rng.choice(len(hardness), n_target, replace=replace or n_target > len(hardness))
    pops, edges = np.histogram(hardness, bins=k_bins)
    bins, contrib = [], np.zeros(k_bins)
    for b in range(k_bins):
        upper = hardness <= edges[b + 1] if b == k_bins - 1 else hardness < edges[b + 1]
        idx = np.where((hardness >= edges[b]) & upper)[0]
        bins.append(idx)
        if len(idx):
            contrib[b] = hardness[idx].mean()
    with np.errstate(divide="ignore"):
        w = 1.0 / (contrib + alpha)
    w[(pops == 0) | ~np.isfinite(w)] = 0.0
    n_bins = np.floor(n_target * w / w.sum()).astype(int)
    order, i = np.argsort(-pops), 0
    while n_bins.sum() < n_target:
        if pops[order[i % k_bins]] > 0:
            n_bins[order[i % k_bins]] += 1
        i += 1
    parts = [rng.choice(bins[b], n_bins[b], replace=replace or n_bins[b] > pops[b])
             for b in range(k_bins) if n_bins[b] > 0]
    return np.concatenate(parts)


class SelfPacedEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, n_estimators=50, k_bins=5, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        counts = [np.sum(y == c) for c in self.classes_]
        self.minority_ = self.classes_[int(np.argmin(counts))]
        X_min, X_maj = X[y == self.minority_], X[y != self.minority_]
        y_min, y_maj = y[y == self.minority_], y[y != self.minority_]
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier()
        hard = np.zeros(len(X_maj))  # running P(minority) of majority samples
        self.estimators_ = []
        for i in range(self.n_estimators):
            alpha = np.tan(np.pi * 0.5 * i / max(self.n_estimators - 1, 1))
            idx = self_paced_sample(hard, len(X_min), alpha, rng, self.k_bins)
            est = clone(base).set_params(random_state=rng.randint(2**31 - 1))
            est.fit(np.vstack([X_min, X_maj[idx]]), np.r_[y_min, y_maj[idx]])
            self.estimators_.append(est)
            p = est.predict_proba(X_maj)[:, list(est.classes_).index(self.minority_)]
            hard = (hard * i + p) / (i + 1)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
