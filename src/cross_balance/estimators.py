"""Estimator wrappers for use as a CrossBalanceClassifier base estimator."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OMPClassifier(ClassifierMixin, BaseEstimator):
    """Adapts Orthogonal Matching Pursuit -- a sparse linear regressor -- to
    binary classification: fits OMP on a {0, 1}-encoded target and clips
    predictions to [0, 1] to use as a probability surrogate.

    OMP greedily selects a small number of nonzero coefficients, which gives
    it a different inductive bias than logistic regression's "shrink
    everything" L2 regularization: on a small, high-dimensional chunk where
    most features are noise, OMP's sparsity can generalize better than a
    dense linear fit. Not a probabilistic model -- ``predict_proba`` is a
    clipped-regression surrogate, not a calibrated probability.
    """

    def __init__(self, n_nonzero_coefs=None, random_state=None):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("OMPClassifier only supports binary classification.")

        max_nonzero = max(1, min(X.shape[1], X.shape[0] - 1))
        n_nonzero = self.n_nonzero_coefs or max(1, min(X.shape[1] // 2, 10))
        n_nonzero = min(n_nonzero, max_nonzero)

        self.omp_ = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        y_float = (y == self.classes_[1]).astype(float)
        self.omp_.fit(X, y_float)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "omp_")
        X = check_array(X)
        p1 = np.clip(self.omp_.predict(X), 0.0, 1.0)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
