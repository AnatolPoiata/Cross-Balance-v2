"""Out-of-fold performance weighting for ensemble members."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold


def oof_score(estimator, X, y, n_splits=3, random_state=None, scorer=average_precision_score):
    """Score ``estimator`` on (X, y) using out-of-fold predictions, so the
    weight reflects generalization on this chunk rather than training fit.
    Falls back to 1.0 (uniform) when there isn't enough data/classes to
    cross-validate internally.
    """
    if n_splits < 2 or len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return 1.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in kf.split(X):
        if len(np.unique(y[train_idx])) < 2:
            return 1.0
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        oof_proba[val_idx] = est.predict_proba(X[val_idx])[:, 1]
    return float(scorer(y, oof_proba))


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones(len(weights)) / len(weights)
    return weights / total
