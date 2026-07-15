"""CrossBalanceClassifier: a chunked-partition ensemble for imbalanced
binary classification.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._partition import bagging_chunks, weighted_chunks
from ._weighting import normalize_weights, oof_score
from .estimators import OMPClassifier


class CrossBalanceClassifier(ClassifierMixin, BaseEstimator):
    """Ensemble classifier for imbalanced binary classification.

    Partitions the majority class into ``n_chunks`` balanced-ratio slices,
    each paired with the full minority class, and trains one clone of
    ``base_estimator`` per slice. Predictions are combined via a weighted
    average of ``predict_proba``.

    Defaults (``base_estimator='auto'``, ``ensemble_type='boosting'``,
    ``n_chunks=50``) reflect the best configuration found across 5 rounds of
    benchmarking against 10 established imbalance-handling methods on 3
    synthetic and 8 real-world datasets (see the project's ``results/``
    directory) — pass explicit values to override.

    Parameters
    ----------
    base_estimator : estimator, ``'auto'``, or None, default=``'auto'``
        Cloned and fit once per chunk. ``'auto'`` picks per chunk from a
        5-tier capacity ladder based on the chunk's samples-per-feature
        density (see ``auto_density_thresholds``), lowest-capacity first:
        ``OMPClassifier`` (sparse linear, for small/high-dimensional chunks
        where dense models overfit) → ``LogisticRegression`` → shallow
        ``DecisionTreeClassifier(max_depth=6)`` → ``RandomForestClassifier``
        (20 trees, depth 8, 30% row subsampling) → ``GradientBoostingClassifier``
        (15 estimators, depth 2, 30% row subsampling) for the densest chunks
        -- the RF/GB configs are intentionally lightweight since ``'auto'``
        may refit them ``n_chunks`` times per ``fit()`` call. ``None`` forces
        ``LogisticRegression(max_iter=1000)`` always; pass any other
        estimator to use it unconditionally.
    auto_density_thresholds : tuple of 4 floats, default=(5.0, 10.0, 30.0, 100.0)
        Only used when ``base_estimator='auto'``. Samples-per-feature
        density boundaries between the 5 tiers above (OMP below the first
        value, GradientBoosting at or above the last).
    ensemble_type : {'bagging', 'boosting'}, default='boosting'
        'bagging' draws chunks by sequential wrap-around sampling of the
        majority class. 'boosting' draws chunks by weighted resampling,
        upweighting majority samples the ensemble-so-far misclassifies
        (cost-sensitive boosting).
    ratio : float, default=2.0
        Majority:minority ratio within each chunk (e.g. 2.0 pairs the full
        minority class with 2x as many majority samples per chunk).
    n_chunks : int, default=50
        Number of chunks / ensemble members.
    weighting : {'uniform', 'oof'}, default='oof'
        'oof' weights each member by its out-of-fold average-precision score
        on its own chunk; 'uniform' averages all members equally.
    internal_splits : int, default=3
        KFold splits used to compute OOF weights when ``weighting='oof'``.
    max_chunk_size : int or None, default=None
        Caps the majority-sample count per chunk (after applying ``ratio``),
        independent of ``n_min``/``n_maj``. ``None`` leaves chunks unbounded
        (the original behavior). Matters most with ``base_estimator='auto'``
        on large or near-balanced datasets: without a cap, a chunk can grow
        to tens of thousands of rows, and the RandomForest/GradientBoosting
        tiers become impractically slow when refit ``n_chunks`` times per
        ``fit()`` call (plus ``internal_splits`` more for OOF weighting).
    random_state : int, RandomState instance or None, default=None
    """

    def __init__(self, base_estimator="auto", auto_density_thresholds=(5.0, 10.0, 30.0, 100.0),
                 ensemble_type="boosting", ratio=2.0, n_chunks=50, weighting="oof",
                 internal_splits=3, max_chunk_size=None, random_state=None):
        self.base_estimator = base_estimator
        self.auto_density_thresholds = auto_density_thresholds
        self.ensemble_type = ensemble_type
        self.ratio = ratio
        self.n_chunks = n_chunks
        self.weighting = weighting
        self.internal_splits = internal_splits
        self.max_chunk_size = max_chunk_size
        self.random_state = random_state

    def _resolve_base_estimator(self, n_chunk_samples, n_features):
        if isinstance(self.base_estimator, str) and self.base_estimator == "auto":
            density = n_chunk_samples / max(n_features, 1)
            t_omp, t_lr, t_tree, t_rf = self.auto_density_thresholds
            rs = self.random_state
            if density < t_omp:
                return OMPClassifier(random_state=rs)
            if density < t_lr:
                return LogisticRegression(max_iter=1000, random_state=rs)
            if density < t_tree:
                return DecisionTreeClassifier(max_depth=6, random_state=rs)
            if density < t_rf:
                return RandomForestClassifier(n_estimators=20, max_depth=8, max_samples=0.3,
                                               random_state=rs, n_jobs=1)
            return GradientBoostingClassifier(n_estimators=15, max_depth=2, subsample=0.3, random_state=rs)
        if self.base_estimator is None:
            return LogisticRegression(max_iter=1000)
        return self.base_estimator

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("CrossBalanceClassifier only supports binary classification.")
        if self.ensemble_type not in ("bagging", "boosting"):
            raise ValueError("ensemble_type must be 'bagging' or 'boosting'")

        rng = check_random_state(self.random_state)

        # Minority = smaller-count class; majority = the other.
        counts = {c: int(np.sum(y == c)) for c in self.classes_}
        minority_class, majority_class = sorted(self.classes_, key=lambda c: counts[c])

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]
        n_min, n_maj = len(X_min), len(X_maj)

        chunk_size = min(max(1, int(round(self.ratio * n_min))), n_maj)

        # Cap the total per-chunk sample count. Chunk composition is
        # min(all n_min minority) + chunk_size majority -- for large/
        # near-balanced datasets n_min alone can be huge, so both sides are
        # scaled down proportionally (preserving `ratio`) rather than only
        # capping the majority side.
        n_min_use = n_min
        if self.max_chunk_size is not None and (n_min + chunk_size) > self.max_chunk_size:
            scale = self.max_chunk_size / (n_min + chunk_size)
            n_min_use = max(1, int(round(n_min * scale)))
            chunk_size = max(1, self.max_chunk_size - n_min_use)

        base = self._resolve_base_estimator(n_min_use + chunk_size, X.shape[1])
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

            if n_min_use < n_min:
                min_idx = check_random_state(seed + i).choice(n_min, size=n_min_use, replace=False)
                X_min_chunk = X_min[min_idx]
            else:
                X_min_chunk = X_min

            X_chunk = np.vstack([X_min_chunk, X_maj[chunk]])
            y_chunk = np.concatenate([
                np.full(n_min_use, minority_class),
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
