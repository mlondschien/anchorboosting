import numpy as np
import pytest

from anchorboost.anchor_loss import AnchorClassificationLoss, AnchorRegressionLoss
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0.1, 1, 10, 100, 1000])
def test_regression_grad(gamma):
    loss = AnchorRegressionLoss(gamma)
    _, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=0.1 / (1 + gamma))


@pytest.mark.parametrize("gamma", [0.1, 1, 10, 100, 1000])
def test_classification_grad(gamma):
    loss = AnchorClassificationLoss(gamma)
    _, y, a = simulate(f2)
    y = (y >= 0).astype(np.int64) + (y >= 1).astype(np.int64)
    rng = np.random.RandomState(0)
    f = rng.normal(size=(len(y), len(loss.init_score(y))))
    check_gradient(loss.loss, loss.grad, f, y, a, stepsize=0.1 / (1 + gamma))


def check_gradient(loss, grad, f, y, anchor, stepsize):
    before = loss(f, y, anchor).sum()
    for i in range(1000):
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
    indices = loss._indices(y, 5)

    array = np.zeros((4, 5))
    for i in range(4):
        array[i, y[i]] = i

    np.testing.assert_equal(array[indices], np.arange(4))


def test_proj():
    loss = AnchorRegressionLoss(1)
    _, y, a = simulate(f2)
    residuals = y - loss.init_score(y)
    np.testing.assert_almost_equal(
        loss._proj_matrix(a) @ residuals, loss._proj(a, residuals)
    )
