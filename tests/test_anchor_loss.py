import numpy as np
import pytest

from anchorboost.anchor_loss import AnchorL2Loss
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 10, 100])
@pytest.mark.parametrize("sigma", [0.1, 0.5])
def test_grad(gamma, sigma):
    loss = AnchorL2Loss(gamma, [0, 1])
    x, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    residuals = sigma * rng.normal(size=x.shape[0])

    assert check_gradient(loss.loss, loss.grad, residuals, a) < 1e-4


def check_gradient(func, grad, residuals, anchor, eps=1e-8):
    approx = (func(residuals + eps, anchor) - func(residuals, anchor)) / eps
    return (approx - grad(residuals, anchor)).mean()


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 10, 100])
def test_hess(gamma):
    loss = AnchorL2Loss(gamma, [0, 1])
    x, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    residuals = 0.1 * rng.normal(size=x.shape[0])
    direction = rng.normal(size=x.shape[0])

    assert check_hessian(loss.grad, loss.hess, residuals, direction, a) < 1e-4


def check_hessian(grad, hess, residuals, direction, anchor, eps=1e-8):
    approx = (grad(residuals + eps * direction, anchor) - grad(residuals, anchor)) / eps
    return (approx - hess(residuals, anchor) @ direction).mean()


def test_proj():
    loss = AnchorL2Loss(1, [0, 1])
    _, y, a = simulate(f2)
    residuals = y - loss.init_score(y)
    np.testing.assert_almost_equal(
        loss._proj_matrix(a) @ residuals, loss._proj(a, residuals)
    )
