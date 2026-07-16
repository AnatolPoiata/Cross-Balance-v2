import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from cross_balance import CrossBalanceClassifier, OMPClassifier


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=8,
        weights=[0.97, 0.03], flip_y=0, random_state=0,
    )
    return X, y


@pytest.mark.parametrize("ensemble_type", ["bagging", "boosting"])
@pytest.mark.parametrize("weighting", ["uniform", "oof"])
def test_fit_predict_proba_shape_and_normalization(imbalanced_data, ensemble_type, weighting):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(
        base_estimator=LogisticRegression(max_iter=500),
        ensemble_type=ensemble_type, ratio=2, n_chunks=5,
        weighting=weighting, random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    assert np.isclose(clf.weights_.sum(), 1.0)


def test_predict_returns_known_classes(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(n_chunks=5, random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= set(clf.classes_)


def test_sklearn_clone_and_get_set_params(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(ratio=3, n_chunks=7, random_state=1)
    clf2 = clone(clf)
    assert clf2.get_params() == clf.get_params()
    clf2.fit(X, y)
    assert len(clf2.estimators_) == 7


def test_base_estimator_none_forces_logistic_regression(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(base_estimator=None, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], LogisticRegression)


def test_defaults_match_validated_config():
    # Regression guard: these defaults were chosen by iterative
    # benchmarking (see results/iteration_*) -- a silent drift here would
    # silently change what "CrossBalanceClassifier()" means.
    params = CrossBalanceClassifier().get_params()
    assert params["base_estimator"] == "auto"
    assert params["ensemble_type"] == "boosting"
    assert params["n_chunks"] == 50
    assert params["ratio"] == 2.0
    assert params["weighting"] == "oof"
    # Independently tuned in iteration 7 (see benchmarks/tune_thresholds.py
    # and results/iteration_7) against a held-out synthetic tuning suite,
    # then validated on the full 11-dataset benchmark -- raised the
    # outright-win count from 5 to 6 of 11 vs. iteration 5's ad hoc-free
    # 2-tier ladder, and from iteration 6's own ad hoc (5,10,30,100) tiers
    # (which regressed 6 of 11 datasets).
    assert params["auto_density_thresholds"] == (2.9, 12.5, 19.3, 238.6)
    assert params["max_chunk_size"] == 3000


def test_rejects_non_binary_targets():
    X = np.random.RandomState(0).randn(30, 4)
    y = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError):
        CrossBalanceClassifier(n_chunks=3).fit(X, y)


def test_works_with_cross_val_score(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(n_chunks=3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=3, scoring="average_precision")
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))


def test_string_labels_supported():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 5)
    y = np.array(["yes"] * 20 + ["no"] * 380)
    rng.shuffle(y)
    clf = CrossBalanceClassifier(n_chunks=3, random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert set(pred) <= {"yes", "no"}
    assert clf.minority_class_ == "yes"


def test_auto_base_estimator_picks_tree_for_mid_density_chunks(imbalanced_data):
    X, y = imbalanced_data  # 2000 rows, 10 features, ~3% minority -> density ~18 (tree tier)
    clf = CrossBalanceClassifier(base_estimator="auto", n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], DecisionTreeClassifier)


# iteration 6's original ad hoc thresholds -- superseded as the default by
# iteration 7's independently-tuned (2.9, 12.5, 19.3, 238.6), but still a
# convenient fixed point for exercising each tier of the ladder in isolation.
_ITERATION_6_ADHOC_TIERS = (5.0, 10.0, 30.0, 100.0)


def test_auto_base_estimator_picks_omp_for_lowest_density_chunks():
    # 60 rows, 30 features, tiny minority -> density ~0.4, well under the OMP tier's threshold
    rng = np.random.RandomState(0)
    X = rng.randn(60, 30)
    y = np.array([1] * 6 + [0] * 54)
    clf = CrossBalanceClassifier(base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
                                  ratio=1, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], OMPClassifier)


def test_auto_base_estimator_picks_logreg_for_low_density_chunks():
    # 200 rows, 6 features, 15 minority, ratio=2 -> chunk density 45/6=7.5 (LR tier: [5, 10))
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    y = np.array([1] * 15 + [0] * 185)
    clf = CrossBalanceClassifier(base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
                                  ratio=2, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], LogisticRegression)


def test_auto_base_estimator_picks_random_forest_for_denser_chunks():
    # 3000 rows, 10 features, 200 minority, ratio=2 -> chunk density 600/10=60 (RF tier: [30, 100))
    rng = np.random.RandomState(0)
    X = rng.randn(3000, 10)
    y = np.array([1] * 200 + [0] * 2800)
    clf = CrossBalanceClassifier(base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
                                  ratio=2, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], RandomForestClassifier)


def test_auto_base_estimator_picks_gradient_boosting_for_densest_chunks():
    # 5000 rows, 5 features, 500 minority, ratio=2 -> chunk density 1500/5=300 (GB tier: >=100)
    rng = np.random.RandomState(0)
    X = rng.randn(5000, 5)
    y = np.array([1] * 500 + [0] * 4500)
    clf = CrossBalanceClassifier(base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
                                  ratio=2, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], GradientBoostingClassifier)


def test_default_thresholds_reach_omp_for_very_sparse_chunks():
    # Unlike iteration 6's ad hoc thresholds, the tuned default (2.9, ...)
    # has a strictly positive OMP boundary, so very sparse chunks now
    # legitimately reach OMP under the *shipped default*, not just when
    # explicitly opted into a wider ladder.
    rng = np.random.RandomState(0)
    X = rng.randn(60, 30)
    y = np.array([1] * 6 + [0] * 54)
    clf = CrossBalanceClassifier(base_estimator="auto", ratio=1, n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], OMPClassifier)


def test_max_chunk_size_caps_total_samples_per_chunk_estimator():
    # Balanced (5000/5000), 5 features -> uncapped chunk would be huge
    # (n_min + ratio*n_min = 5000 + 10000 = 15000), very high density -> GB tier.
    # With max_chunk_size=200, density should drop to a lower tier and the
    # estimator actually used should reflect a much smaller effective chunk.
    rng = np.random.RandomState(0)
    X = rng.randn(10000, 5)
    y = np.array([1] * 5000 + [0] * 5000)
    clf_uncapped = CrossBalanceClassifier(base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
                                           ratio=2, n_chunks=2, max_chunk_size=None, random_state=0).fit(X, y)
    clf_capped = CrossBalanceClassifier(
        base_estimator="auto", auto_density_thresholds=_ITERATION_6_ADHOC_TIERS,
        ratio=2, n_chunks=2, max_chunk_size=200, random_state=0,
    ).fit(X, y)
    assert isinstance(clf_uncapped.estimators_[0], GradientBoostingClassifier)
    assert not isinstance(clf_capped.estimators_[0], GradientBoostingClassifier)


def test_max_chunk_size_default_is_3000():
    clf = CrossBalanceClassifier(n_chunks=3, random_state=0)
    assert clf.max_chunk_size == 3000


def test_max_chunk_size_none_is_explicitly_unbounded(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(n_chunks=3, max_chunk_size=None, random_state=0)
    assert clf.max_chunk_size is None
    clf.fit(X, y)  # should not raise, behaves as before max_chunk_size existed


def test_recovers_minority_class_reasonably_well(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(
        base_estimator=LogisticRegression(max_iter=500),
        n_chunks=10, ratio=2, weighting="oof", random_state=0,
    ).fit(X, y)
    pred = clf.predict(X)
    recall = (pred[y == 1] == 1).mean()
    assert recall > 0.5
