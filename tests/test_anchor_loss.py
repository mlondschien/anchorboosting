import numpy as np
import pytest

from anchorboost.anchor_loss import AnchorL2Loss
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 10, 100])
@pytest.mark.parametrize("sigma", [0.1, 0.5])
def test_grad(gamma, sigma):
    loss = AnchorL2Loss(gamma)
    x, y, a = simulate(f2)
    rng = np.random.RandomState(0)
    residuals = sigma * rng.normal(size=x.shape[0])

    assert check_gradient(loss.anchor_loss, loss.grad, residuals, a) < 1e-4


def check_gradient(func, grad, residuals, anchor, eps=1e-8):
    approx = (func(residuals + eps, anchor) - func(residuals, anchor)) / eps
    return (approx - grad(residuals, anchor)).mean()


def test_proj():
    loss = AnchorL2Loss(1)
    _, y, a = simulate(f2)
    residuals = y - loss.init_score(y)
    np.testing.assert_almost_equal(
        loss._proj_matrix(a) @ residuals, loss._proj(a, residuals)
    )
