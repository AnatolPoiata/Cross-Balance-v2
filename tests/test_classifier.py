import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cross_balance import CrossBalanceClassifier


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


def test_default_base_estimator_is_logistic_regression(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(n_chunks=3, random_state=0).fit(X, y)
    assert isinstance(clf.estimators_[0], LogisticRegression)


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


def test_recovers_minority_class_reasonably_well(imbalanced_data):
    X, y = imbalanced_data
    clf = CrossBalanceClassifier(
        base_estimator=LogisticRegression(max_iter=500),
        n_chunks=10, ratio=2, weighting="oof", random_state=0,
    ).fit(X, y)
    pred = clf.predict(X)
    recall = (pred[y == 1] == 1).mean()
    assert recall > 0.5
