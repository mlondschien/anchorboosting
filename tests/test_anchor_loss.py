import numpy as np
import pytest

from anchorboost.anchor_loss import AnchorClassificationLoss, AnchorRegressionLoss
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_regression_grad(gamma):
    loss = AnchorRegressionLoss(gamma)
    _, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=1 / (1 + gamma))


@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_classification_grad(gamma):
    loss = AnchorClassificationLoss(gamma)
    _, y, a = simulate(f2)
    y = (y >= 0).astype(np.int64) + (y >= 1).astype(np.int64)
    rng = np.random.RandomState(0)
    f = rng.normal(size=(len(y), len(loss.init_score(y))))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=1 / (1 + gamma))


def check_gradient(loss, grad, f, y, anchor, stepsize):
    before = loss(f, y, anchor).sum()
    for i in range(200):
        f = f - stepsize * grad(f, y, anchor)
        after = loss(f, y, anchor).sum()
        if not after - before <= 1e-14:
            raise ValueError(
                f"Gradient is not decreasing at step {i}: {after} > {before}"
            )
        before = after


def test_indices():
    loss = AnchorClassificationLoss(1)
    y = np.array([1, 3, 2, 2])
    indices = loss._indices(y)

    array = np.zeros((4, 5))
    for i in range(4):
        array[i, y[i]] = i

    np.testing.assert_equal(array[indices], np.arange(4))


@pytest.mark.parametrize(
    "anchor, residuals, result",
    [
        (
            np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0]]),
            np.array([[1], [2], [3]]),
            np.array([[1.5], [1.5], [3]]),
        ),
        (
            np.array([[2], [2], [1], [3]]),
            np.array([[1], [2], [3], [1]]),
            np.array([[1.5], [1.5], [3], [1]]),
        ),
        (
            np.array([[1.0], [1.0], [0.0]]),
            np.array([[1, 2], [0, 2], [3, 4]]),
            np.array([[0.5, 2.0], [0.5, 2.0], [0.0, 0.0]]),
        ),
        (
            np.array([[2], [2], [1]]),
            np.array([[1, 2], [0, 2], [3, 4]]),
            np.array([[0.5, 2.0], [0.5, 2.0], [3.0, 4.0]]),
        ),
        (np.array([[0.0]]), np.array([[1]]), np.array([[0]])),
    ],
)
def test_proj(anchor, residuals, result):
    loss = AnchorRegressionLoss(1)
    np.testing.assert_almost_equal(loss._proj(anchor, residuals), result)
