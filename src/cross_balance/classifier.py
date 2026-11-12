"""CrossBalanceClassifier: a chunked-partition ensemble for imbalanced
binary classification.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._partition import bagging_chunks, self_paced_chunk, weighted_chunks
from ._weighting import normalize_weights, oof_score
from .estimators import OMPClassifier

# The v0.1 (iteration-7) configuration, kept so earlier results stay
# reproducible: CrossBalanceClassifier(**LEGACY_V2_PARAMS).
LEGACY_V2_PARAMS = dict(
    base_estimator="auto", auto_density_thresholds=(2.9, 12.5, 19.3, 238.6),
    ensemble_type="boosting", ratio=2.0, n_chunks=50, weighting="oof",
    internal_splits=3, max_chunk_size=3000, calibration=None, decision_threshold=0.5,
)


def _hgb(rs):
    return HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, max_leaf_nodes=15,
                                          early_stopping=False, random_state=rs)


def _et(rs):
    return ExtraTreesClassifier(n_estimators=20, min_samples_leaf=2, n_jobs=1, random_state=rs)


class CrossBalanceClassifier(ClassifierMixin, BaseEstimator):
    """Ensemble classifier for imbalanced binary classification.

    Draws ``n_chunks`` balanced-ratio samples of the majority class, pairs
    each with the minority class, trains one base estimator per chunk and
    averages their scores.

    Defaults (v0.2) are the configuration frozen after the v3 development
    grid on the development datasets only (see ``article/``): self-paced
    hardness sampling, a mixed HistGradientBoosting / ExtraTrees ensemble,
    ``ratio=4``, ``n_chunks=50``, ``max_chunk_size=3000``, uniform member
    weights and prior-corrected probabilities. The v0.1 configuration is
    available as ``CrossBalanceClassifier(**LEGACY_V2_PARAMS)``.

    Parameters
    ----------
    base_estimator : estimator, ``'mixed'``, ``'hgb'``, ``'auto'`` or None, default=``'mixed'``
        ``'mixed'`` alternates a small ``HistGradientBoostingClassifier``
        (50 iterations, 15 leaves) and an ``ExtraTreesClassifier`` (20
        trees) between consecutive chunks -- two low-correlation learners
        that add diversity to the ensemble. ``'hgb'`` uses only the former.
        ``'auto'`` is the v0.1 5-tier density ladder
        (OMP / LR / tree / RF / GB, see ``auto_density_thresholds``),
        selected once from the capped chunk size. ``None`` forces
        ``LogisticRegression(max_iter=1000)``; any other estimator is
        cloned and used unconditionally.
    auto_density_thresholds : tuple of 4 floats, default=(2.9, 12.5, 19.3, 238.6)
        Only used when ``base_estimator='auto'``: samples-per-feature
        boundaries of the v0.1 ladder (computed on the capped chunk size).
    ensemble_type : {'self_paced', 'boosting', 'bagging'}, default='self_paced'
        How the majority sample of each chunk is drawn.
        ``'self_paced'`` bins majority samples by the ensemble-so-far
        probability of the minority class and samples bins with
        hardness-harmonised quotas (Liu et al., 2020), without replacement
        inside a bin. ``'boosting'`` (v0.1) resamples with replacement with
        weights doubled every time a single member misclassifies a sample;
        weights compound geometrically, so late chunks are dominated by a
        few duplicated borderline/noisy samples. ``'bagging'`` draws
        sequential wrap-around slices of a shuffled majority class.
    ratio : float, default=4.0
        Majority:minority ratio within each chunk.
    n_chunks : int, default=50
        Number of chunks / ensemble members.
    weighting : {'uniform', 'oof'}, default='uniform'
        ``'oof'`` (v0.1) weights members by internal out-of-fold average
        precision on their own chunk; on the development grid the weights
        were nearly uniform (0.017-0.026 for 50 members) at 4x the fitting
        cost, so v0.2 averages members uniformly.
    internal_splits : int, default=3
        KFold splits for ``weighting='oof'``.
    max_chunk_size : int or None, default=3000
        Caps the total per-chunk sample count; above it both classes are
        scaled down proportionally (preserving ``ratio``).
    calibration : {'prior', None}, default='prior'
        ``'prior'`` maps the averaged chunk score ``s`` back to the
        training prevalence, ``p = k*s / (k*s - s + 1)`` with
        ``k = beta / gamma`` the ratio of the majority and minority
        sampling rates (Dal Pozzolo et al., 2015, generalised to a
        subsampled minority). Monotone, so it never changes rankings
        (AP, ROC-AUC, Precision@k); it makes ``predict_proba`` usable for
        cost-based decisions. ``None`` returns the raw chunk-prevalence
        scores (v0.1 behaviour).
    decision_threshold : float or 'f1', default=0.5
        Threshold applied to ``predict_proba[:, minority]`` by
        ``predict``. ``'f1'`` estimates the F1-maximising threshold from
        internal 3-fold out-of-fold scores (3 extra ensemble fits).
        For a cost matrix, pass ``c_fp / (c_fp + c_fn)`` together with
        ``calibration='prior'``.
    random_state : int, RandomState instance or None, default=None
    """

    def __init__(self, base_estimator="mixed",
                 auto_density_thresholds=(2.9, 12.5, 19.3, 238.6),
                 ensemble_type="self_paced", ratio=4.0, n_chunks=50, weighting="uniform",
                 internal_splits=3, max_chunk_size=3000, calibration="prior",
                 decision_threshold=0.5, random_state=None):
        self.base_estimator = base_estimator
        self.auto_density_thresholds = auto_density_thresholds
        self.ensemble_type = ensemble_type
        self.ratio = ratio
        self.n_chunks = n_chunks
        self.weighting = weighting
        self.internal_splits = internal_splits
        self.max_chunk_size = max_chunk_size
        self.calibration = calibration
        self.decision_threshold = decision_threshold
        self.random_state = random_state

    def _resolve_base_estimator(self, n_chunk_samples, n_features, member=0, seed=None):
        rs = self.random_state if seed is None else seed
        if isinstance(self.base_estimator, str):
            if self.base_estimator == "mixed":
                return _hgb(rs) if member % 2 == 0 else _et(rs)
            if self.base_estimator == "hgb":
                return _hgb(rs)
            if self.base_estimator == "auto":
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
                return GradientBoostingClassifier(n_estimators=15, max_depth=2, subsample=0.3,
                                                  random_state=rs)
            raise ValueError(f"Unknown base_estimator '{self.base_estimator}'")
        if self.base_estimator is None:
            return LogisticRegression(max_iter=1000)
        return self.base_estimator

    def _validate_params_values(self):
        if self.ensemble_type not in ("self_paced", "bagging", "boosting"):
            raise ValueError("ensemble_type must be 'self_paced', 'bagging' or 'boosting'")
        if self.weighting not in ("uniform", "oof"):
            raise ValueError("weighting must be 'uniform' or 'oof'")
        if self.calibration not in ("prior", None):
            raise ValueError("calibration must be 'prior' or None")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("CrossBalanceClassifier only supports binary classification.")
        self._validate_params_values()

        if self.decision_threshold == "f1":
            self.threshold_ = self._estimate_f1_threshold(X, y)
        else:
            self.threshold_ = float(self.decision_threshold)
        return self._fit_ensemble(X, y)

    def _estimate_f1_threshold(self, X, y):
        counts = {c: int(np.sum(y == c)) for c in self.classes_}
        minority = min(self.classes_, key=lambda c: counts[c])
        n_splits = int(min(3, counts[minority]))
        if n_splits < 2:
            return 0.5
        scores = np.zeros(len(y))
        inner = clone(self).set_params(decision_threshold=0.5)
        cv = StratifiedKFold(n_splits, shuffle=True,
                             random_state=check_random_state(self.random_state).randint(2**31 - 1))
        for tr, va in cv.split(X, y):
            m = clone(inner)._fit_ensemble(X[tr], y[tr])
            scores[va] = m.predict_proba(X[va])[:, list(m.classes_).index(minority)]
        precision, recall, thresholds = precision_recall_curve(y == minority, scores)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return float(thresholds[int(np.argmax(f1[:-1]))])

    def _fit_ensemble(self, X, y):
        self.classes_ = unique_labels(y)
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

        # Sampling rates of each class, used to undo the chunk prevalence
        # shift in predict_proba (calibration='prior').
        self.majority_rate_ = chunk_size / n_maj
        self.minority_rate_ = n_min_use / n_min
        self.chunk_size_ = (n_min_use, chunk_size)

        seed = rng.randint(0, np.iinfo(np.int32).max)

        if self.ensemble_type == "bagging":
            chunk_index_list = bagging_chunks(n_maj, chunk_size, self.n_chunks, random_state=seed)
        elif self.ensemble_type == "boosting":
            maj_sample_weights = np.ones(n_maj)
        else:
            maj_hardness = np.zeros(n_maj)  # ensemble-so-far P(minority) on the majority

        estimators, weights = [], []
        for i in range(self.n_chunks):
            member_rng = check_random_state(seed + i)
            if self.ensemble_type == "boosting":
                chunk = weighted_chunks(n_maj, chunk_size, 1, maj_sample_weights,
                                        random_state=seed + i)[0]
            elif self.ensemble_type == "self_paced":
                alpha = np.tan(np.pi * 0.5 * i / max(self.n_chunks - 1, 1))
                chunk = self_paced_chunk(maj_hardness, chunk_size, alpha, member_rng)
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

            base = self._resolve_base_estimator(n_min_use + chunk_size, X.shape[1],
                                                member=i, seed=seed + i)
            estimator = clone(base)
            estimator.fit(X_chunk, y_chunk)
            estimators.append(estimator)

            if self.weighting == "oof":
                w = oof_score(clone(base), X_chunk, (y_chunk == minority_class).astype(int),
                              n_splits=self.internal_splits, random_state=seed + i)
            else:
                w = 1.0
            weights.append(w)

            if self.ensemble_type in ("boosting", "self_paced"):
                classes = list(estimator.classes_)
                proba_min = estimator.predict_proba(X_maj)[:, classes.index(minority_class)]
                if self.ensemble_type == "boosting":
                    # v0.1 cost-sensitive update: double the weight of majority
                    # samples this member misclassifies as minority.
                    misclassified = (proba_min > 0.5).astype(float)
                    maj_sample_weights *= 1.0 + misclassified
                    maj_sample_weights /= maj_sample_weights.sum()
                else:
                    maj_hardness = (maj_hardness * i + proba_min) / (i + 1)

        self.estimators_ = estimators
        self.weights_ = normalize_weights(weights)
        self.minority_class_ = minority_class
        self.majority_class_ = majority_class
        if not hasattr(self, "threshold_"):
            self.threshold_ = 0.5
        return self

    def _chunk_score(self, X):
        """Weighted average of the members' minority-class scores, at the
        chunk prevalence (before prior correction)."""
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        score = np.zeros(X.shape[0])
        for estimator, w in zip(self.estimators_, self.weights_):
            classes = list(estimator.classes_)
            score += w * estimator.predict_proba(X)[:, classes.index(self.minority_class_)]
        return score

    def predict_proba(self, X):
        s = self._chunk_score(X)
        if self.calibration == "prior":
            k = self.majority_rate_ / self.minority_rate_
            s = k * s / (k * s - s + 1.0)
        min_pos = list(self.classes_).index(self.minority_class_)
        out = np.empty((len(s), 2))
        out[:, min_pos] = s
        out[:, 1 - min_pos] = 1.0 - s
        return out

    def predict(self, X):
        p_min = self.predict_proba(X)[:, list(self.classes_).index(self.minority_class_)]
        return np.where(p_min >= self.threshold_, self.minority_class_, self.majority_class_)
