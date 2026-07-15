import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from cross_balance import OMPClassifier


@pytest.fixture
def small_data():
    X, y = make_classification(
        n_samples=200, n_features=15, n_informative=5,
        weights=[0.8, 0.2], flip_y=0, random_state=0,
    )
    return X, y


def test_fit_predict_proba_shape(small_data):
    X, y = small_data
    clf = OMPClassifier(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_predict_returns_known_classes(small_data):
    X, y = small_data
    clf = OMPClassifier(random_state=0).fit(X, y)
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= set(clf.classes_)


def test_clone_compliant(small_data):
    X, y = small_data
    clf = OMPClassifier(n_nonzero_coefs=5, random_state=0)
    clf2 = clone(clf)
    assert clf2.get_params() == clf.get_params()
    clf2.fit(X, y)


def test_n_nonzero_coefs_capped_below_feature_and_sample_count():
    # Fewer samples than requested nonzero coefs shouldn't error.
    rng = np.random.RandomState(0)
    X = rng.randn(5, 20)
    y = np.array([0, 0, 0, 1, 1])
    clf = OMPClassifier(n_nonzero_coefs=50, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (5, 2)


def test_rejects_non_binary_targets():
    X = np.random.RandomState(0).randn(30, 4)
    y = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError):
        OMPClassifier().fit(X, y)
