import lightgbm as lgb
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from anchorboosting.objectives.mixins import (
    ClassificationMixin,
    MultiClassificationMixin,
)
from anchorboosting.simulate import f2, simulate


def test_classification_mixin():
    loss = ClassificationMixin()
    X, y, _ = simulate(f2, n=10)
    y = (y > 0).astype(int)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    data = lgb.Dataset(X, y)

    assert (loss.loss(f, data) >= 0).all()  # loss is non-negative

    grad_approx = approx_fprime(f, lambda f_: loss.loss(f_, data).sum(), 1e-6)
    grad = loss.grad(f, data)
    np.testing.assert_allclose(grad_approx, grad, rtol=1e-5, atol=1e-6)

    hess_approx = np.diag(approx_fprime(f, lambda f_: loss.grad(f_, data), 1e-5))
    hess = loss.hess(f, data)
    np.testing.assert_allclose(hess_approx, hess, rtol=5e-4, atol=5e-4)


def test_multi_classification_mixin():
    loss = MultiClassificationMixin(n_classes=3)
    X, y, _ = simulate(f2, n=10)
    y = (y > 0).astype(int) + (y > 1).astype(int)
    rng = np.random.RandomState(0)
    f = rng.normal(size=3 * len(y))
    data = lgb.Dataset(X, y)

    assert (loss.loss(f, data) >= 0).all()  # loss is non-negative

    grad_approx = approx_fprime(f, lambda f_: loss.loss(f_, data).sum(), 1e-6)
    grad = loss.grad(f, data)
    np.testing.assert_allclose(grad_approx, grad, rtol=1e-5, atol=1e-6)

    hess_approx = np.diag(approx_fprime(f, lambda f_: loss.grad(f_, data), 1e-5))
    hess = loss.hess(f, data)
    np.testing.assert_allclose(hess_approx, loss.factor * hess, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("y", [[0, 1, 3, 2, 2], [1, 1, 1, 0, 1]])
def test_indices(y):
    n_unique = len(np.unique(y))
    loss = MultiClassificationMixin(n_unique)
    y = np.array(y)
    indices = loss._indices(y)

    array = np.zeros((len(y), n_unique))
    for i in range(len(y)):
        array[i, y[i]] = i

    np.testing.assert_equal(array[indices], np.arange(len(y)))


@pytest.mark.parametrize(
    "y, f",
    [
        ([0, 1, 2, 2], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        ([0, 1], [[0, 0], [0, 0]]),
    ],
)
def test_negative_log_likelihood_classification(y, f):
    loss = MultiClassificationMixin(len(np.unique(y)))
    y = np.array(y)
    f = np.array(f)
    data = lgb.Dataset(np.ones(len(y)), y)
    np.testing.assert_almost_equal(
        -loss.loss(f, data),
        np.log(loss.predictions(f)[loss._indices(y)]),
    )


@pytest.mark.parametrize(
    "y", [[0, 1, 2, 2], [0, 1], [0, 1, 2, 3, 4, 5, 1, 1, 1, 2, 3, 5]]
)
def test_init_scores_classification(y):
    unique_values, unique_counts = np.unique(y, return_counts=True)
    expected = np.tile(np.array(unique_counts) / np.sum(unique_counts), (len(y), 1))
    loss = MultiClassificationMixin(len(unique_values))
    init_scores = loss.init_score(y).reshape(len(y), -1, order="F")
    predictions = loss.predictions(init_scores)
    np.testing.assert_almost_equal(predictions, expected)
