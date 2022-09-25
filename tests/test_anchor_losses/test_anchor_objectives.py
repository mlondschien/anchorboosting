import lightgbm as lgb
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from anchorboost.anchor_losses import (
    AnchorKookClassificationObjective,
    AnchorKookMultiClassificationObjective,
    AnchorRegressionObjective,
)
from anchorboost.simulate import f2, simulate


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_anchor_kook_classification_objective(gamma):
    loss = AnchorKookClassificationObjective(gamma=gamma)
    X, y, a = simulate(f2, n=10)
    y = (y > 0).astype(int)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    data = lgb.Dataset(X, y)
    data.anchor = a

    if gamma >= 1:
        assert (loss.loss(f, data) >= 0).all()  # loss is non-negative

    grad_approx = approx_fprime(f, lambda f_: loss.loss(f_, data).sum(), 1e-6)
    grad = loss.grad(f, data)
    np.testing.assert_allclose(grad_approx, grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_anchor_kook_multi_classification_objective(gamma):
    loss = AnchorKookMultiClassificationObjective(n_classes=3, gamma=gamma)
    X, y, a = simulate(f2, n=10)
    y = (y > 0).astype(int) + (y > 1).astype(int)
    rng = np.random.RandomState(0)
    f = rng.normal(size=3 * len(y))
    data = lgb.Dataset(X, y)
    data.anchor = a

    if gamma >= 1:
        assert (loss.loss(f, data) >= 0).all()  # loss is non-negative

    grad_approx = approx_fprime(f, lambda f_: loss.loss(f_, data).sum(), 1e-6)
    grad = loss.grad(f, data)
    np.testing.assert_allclose(grad_approx, grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_anchor_regression_objective(gamma):
    loss = AnchorRegressionObjective(gamma=gamma)
    X, y, a = simulate(f2, n=10)
    rng = np.random.RandomState(0)
    f = rng.normal(size=len(y))
    data = lgb.Dataset(X, y)
    data.anchor = a

    if gamma >= 1:
        assert (loss.loss(f, data) >= 0).all()  # loss is non-negative

    grad_approx = approx_fprime(f, lambda f_: loss.loss(f_, data).sum(), 1e-6)
    grad = loss.grad(f, data)
    np.testing.assert_allclose(grad_approx, grad, rtol=1e-5, atol=1e-6)
