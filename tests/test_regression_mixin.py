import lightgbm as lgb
import numpy as np
from scipy.optimize import approx_fprime

from anchorboosting.objectives.mixins import RegressionMixin
from anchorboosting.simulate import f2, simulate


def test_classification_mixin():
    loss = RegressionMixin()
    X, y, _ = simulate(f2, n=10)

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
