import lightgbm as lgb
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from anchorboosting.objectives.anchor_objectives import (
    AnchorHSICRegressionObjective,
    AnchorKookClassificationObjective,
    AnchorKookMultiClassificationObjective,
    AnchorLiuClassificationObjective,
    AnchorRegressionObjective,
)
from anchorboosting.simulate import f2, simulate


@pytest.mark.parametrize("center_residuals", [True, False])
@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_anchor_kook_classification_objective(gamma, center_residuals):
    loss = AnchorKookClassificationObjective(
        gamma=gamma, center_residuals=center_residuals
    )
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


@pytest.mark.parametrize("center_residuals", [True, False])
@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_anchor_kook_multi_classification_objective(gamma, center_residuals):
    loss = AnchorKookMultiClassificationObjective(
        n_classes=3, gamma=gamma, center_residuals=center_residuals
    )
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
def test_liu_classification_objective(gamma):
    loss = AnchorLiuClassificationObjective(gamma=gamma)
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


@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_hsic_anchor_regression_objective(gamma):
    loss = AnchorHSICRegressionObjective(gamma=gamma)
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


@pytest.mark.parametrize("center_residuals", [True, False])
@pytest.mark.parametrize("gamma", [0, 0.5, 1, 5, 100])
def test_compare_kook_anchor_classifications(gamma, center_residuals):
    single_loss = AnchorKookClassificationObjective(
        gamma=gamma, center_residuals=center_residuals
    )
    multi_loss = AnchorKookMultiClassificationObjective(
        n_classes=2, gamma=gamma, center_residuals=center_residuals
    )

    X, y, a = simulate(f2, n=100)
    y = (y > 0).astype(int)

    single_data = lgb.Dataset(X, y, init_score=single_loss.init_score(y))
    single_data.anchor = a

    multi_data = lgb.Dataset(X, y, init_score=multi_loss.init_score(y))
    multi_data.anchor = a

    single_model = lgb.train(
        params={"learning_rate": 0.1, "objective": single_loss.objective},
        train_set=single_data,
        num_boost_round=10,
    )

    multi_model = lgb.train(
        params={
            "learning_rate": 0.1,
            "num_class": 2,
            "objective": multi_loss.objective,
        },
        train_set=multi_data,
        num_boost_round=10,
    )

    single_pred = single_model.predict(X)
    multi_pred = multi_model.predict(X)

    np.testing.assert_allclose(
        single_pred, multi_pred[:, 1] - multi_pred[:, 0], rtol=1e-5, atol=1e-6
    )
