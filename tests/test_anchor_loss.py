import lightgbm as lgb
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from anchorboost.anchor_loss import AnchorClassificationLoss, AnchorRegressionLoss
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0, 0.1, 0.5, 1, 2, 10, 100, 1000])
def test_objective_regression(gamma):
    loss = AnchorRegressionLoss(gamma)
    X, y, a = simulate(f2, n=100)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    data = lgb.Dataset(X, y)
    data.anchor = a
    obj_approx = approx_fprime(f, lambda f_: len(y) * loss.score(f_, data)[1], 1e-6)
    np.testing.assert_allclose(obj_approx, loss.objective(f, data)[0], rtol=1e-5)


@pytest.mark.parametrize("gamma", [0, 0.1, 0.5, 1, 2, 10, 100, 1000])
def test_objective_classification(gamma):
    loss = AnchorClassificationLoss(gamma, n_classes=3)
    X, y, a = simulate(f2, n=10)
    y = (y > 0).astype(int) + (y > 1).astype(int)
    rng = np.random.RandomState(0)
    f = rng.normal(size=3 * len(y))
    data = lgb.Dataset(X, y)
    data.anchor = a
    obj_approx = approx_fprime(f, lambda f_: len(y) * loss.score(f_, data)[1], 1e-6)
    np.testing.assert_allclose(obj_approx, loss.objective(f, data)[0], rtol=1e-5)


@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_grad_regression(gamma):
    loss = AnchorRegressionLoss(gamma)
    _, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=1 / (gamma + 1))


@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_grad_classification(gamma):
    loss = AnchorClassificationLoss(gamma, n_classes=3)
    _, y, a = simulate(f2)
    y = (y >= 0).astype(np.int64) + (y >= 1).astype(np.int64)
    rng = np.random.RandomState(0)
    f = rng.normal(size=(len(y), loss.n_classes))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=1 / (gamma + 1))


def check_gradient(loss, grad, f, y, anchor, stepsize):
    """Make sure that moving in the direction of the negative gradient reduces the loss.

    Parameters
    ----------
    loss : callable
        Loss function. Arguments are f, y, anchor.
    grad : callable
        Gradient function. Arguments are f, y, anchor.
    f : array-like
        Initial value of f.
    y : array-like
        Target values.
    anchor : array-like
        Anchor values.
    stepsize : float
        Step size for gradient descent.
    """
    before = loss(f, y, anchor).sum()
    for i in range(200):
        f = f - stepsize * grad(f, y, anchor)
        after = loss(f, y, anchor).sum()
        if not after - before <= 1e-14:
            raise ValueError(
                f"Gradient is not decreasing at step {i}: {after} > {before}"
            )
        before = after


@pytest.mark.parametrize("y", [[0, 1, 3, 2, 2], [1, 1, 1, 0, 1]])
def test_indices(y):
    n_unique = len(np.unique(y))
    loss = AnchorClassificationLoss(1, n_unique)
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
    loss = AnchorClassificationLoss(1, len(np.unique(y)))
    y = np.array(y)
    f = np.array(f)

    np.testing.assert_almost_equal(
        -loss.negative_log_likelihood(f, y),
        np.log(loss.predictions(f)[loss._indices(y)]),
    )


@pytest.mark.parametrize(
    "y", [[0, 1, 2, 2], [0, 1], [0, 1, 2, 3, 4, 5, 1, 1, 1, 2, 3, 5]]
)
def test_init_scores_classification(y):
    unique_values, unique_counts = np.unique(y, return_counts=True)
    expected = np.tile(np.array(unique_counts) / np.sum(unique_counts), (len(y), 1))
    loss = AnchorClassificationLoss(1, len(unique_values))
    init_scores = loss.init_score(y).reshape(len(y), -1, order="F")
    predictions = loss.predictions(init_scores)
    np.testing.assert_almost_equal(predictions, expected)
