"""CrossBalanceClassifier: a chunked-partition ensemble for imbalanced
binary classification.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._partition import bagging_chunks, weighted_chunks
from ._weighting import normalize_weights, oof_score


class CrossBalanceClassifier(ClassifierMixin, BaseEstimator):
    """Ensemble classifier for imbalanced binary classification.

    Partitions the majority class into ``n_chunks`` balanced-ratio slices,
    each paired with the full minority class, and trains one clone of
    ``base_estimator`` per slice. Predictions are combined via a weighted
    average of ``predict_proba``.

    Parameters
    ----------
    base_estimator : estimator, default=None
        Cloned and fit once per chunk. Defaults to
        ``LogisticRegression(max_iter=1000)`` if None.
    ensemble_type : {'bagging', 'boosting'}, default='bagging'
        'bagging' draws chunks by sequential wrap-around sampling of the
        majority class. 'boosting' draws chunks by weighted resampling,
        upweighting majority samples the ensemble-so-far misclassifies
        (cost-sensitive boosting).
    ratio : float, default=2.0
        Majority:minority ratio within each chunk (e.g. 2.0 pairs the full
        minority class with 2x as many majority samples per chunk).
    n_chunks : int, default=10
        Number of chunks / ensemble members.
    weighting : {'uniform', 'oof'}, default='oof'
        'oof' weights each member by its out-of-fold average-precision score
        on its own chunk; 'uniform' averages all members equally.
    internal_splits : int, default=3
        KFold splits used to compute OOF weights when ``weighting='oof'``.
    random_state : int, RandomState instance or None, default=None
    """

    def __init__(self, base_estimator=None, ensemble_type="bagging", ratio=2.0,
                 n_chunks=10, weighting="oof", internal_splits=3, random_state=None):
        self.base_estimator = base_estimator
        self.ensemble_type = ensemble_type
        self.ratio = ratio
        self.n_chunks = n_chunks
        self.weighting = weighting
        self.internal_splits = internal_splits
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("CrossBalanceClassifier only supports binary classification.")
        if self.ensemble_type not in ("bagging", "boosting"):
            raise ValueError("ensemble_type must be 'bagging' or 'boosting'")

        rng = check_random_state(self.random_state)
        base = self.base_estimator if self.base_estimator is not None else LogisticRegression(max_iter=1000)

        # Minority = smaller-count class; majority = the other.
        counts = {c: int(np.sum(y == c)) for c in self.classes_}
        minority_class, majority_class = sorted(self.classes_, key=lambda c: counts[c])

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]
        n_min, n_maj = len(X_min), len(X_maj)

        chunk_size = min(max(1, int(round(self.ratio * n_min))), n_maj)
        seed = rng.randint(0, np.iinfo(np.int32).max)

        if self.ensemble_type == "bagging":
            chunk_index_list = bagging_chunks(n_maj, chunk_size, self.n_chunks, random_state=seed)
        else:
            maj_sample_weights = np.ones(n_maj)

        estimators, weights = [], []
        for i in range(self.n_chunks):
            if self.ensemble_type == "boosting":
                chunk = weighted_chunks(n_maj, chunk_size, 1, maj_sample_weights,
                                         random_state=seed + i)[0]
            else:
                chunk = chunk_index_list[i]

            X_chunk = np.vstack([X_min, X_maj[chunk]])
            y_chunk = np.concatenate([
                np.full(n_min, minority_class),
                np.full(len(chunk), majority_class),
            ])

            estimator = clone(base)
            estimator.fit(X_chunk, y_chunk)
            estimators.append(estimator)

            if self.weighting == "oof":
                w = oof_score(clone(base), X_chunk, (y_chunk == minority_class).astype(int),
                               n_splits=self.internal_splits, random_state=seed + i)
            else:
                w = 1.0
            weights.append(w)

            if self.ensemble_type == "boosting":
                # Cost-sensitive update: upweight majority samples this member
                # misclassifies as minority, so later members focus on them.
                classes = list(estimator.classes_)
                proba_min = estimator.predict_proba(X_maj)[:, classes.index(minority_class)]
                misclassified = (proba_min > 0.5).astype(float)
                maj_sample_weights *= 1.0 + misclassified
                maj_sample_weights /= maj_sample_weights.sum()

        self.estimators_ = estimators
        self.weights_ = normalize_weights(weights)
        self.minority_class_ = minority_class
        self.majority_class_ = majority_class
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        n = X.shape[0]
        min_pos = list(self.classes_).index(self.minority_class_)
        maj_pos = list(self.classes_).index(self.majority_class_)

        out = np.zeros((n, 2))
        for estimator, w in zip(self.estimators_, self.weights_):
            classes = list(estimator.classes_)
            p = estimator.predict_proba(X)
            out[:, min_pos] += w * p[:, classes.index(self.minority_class_)]
            out[:, maj_pos] += w * p[:, classes.index(self.majority_class_)]
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
