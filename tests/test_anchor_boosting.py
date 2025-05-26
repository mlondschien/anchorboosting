import lightgbm as lgb
import numpy as np
import pytest
import scipy

from anchorboosting import AnchorBooster
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("gamma", [1.0, 2.0, 100])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_anchor_boosting_second_order(gamma, objective):
    learning_rate = 0.1
    num_leaves = 5
    n = 200
    num_boost_round = 10

    x, y, a = simulate(f1, n=n, shift=0, seed=0)

    if objective == "binary":
        y = (y > 0).astype(int)

    model = AnchorBooster(
        gamma=gamma,
        num_boost_round=num_boost_round,
        num_leaves=num_leaves,
        objective=objective,
        learning_rate=learning_rate,
    )
    model.fit(x, y, Z=a)

    f = model.predict(x, num_iteration=num_boost_round - 1, raw_score=True)

    leaves = model.booster.predict(
        x, pred_leaf=True, start_iteration=num_boost_round - 1, num_iteration=1
    ).flatten()

    def regression_loss(leaf_values):
        residuals = y - f - leaf_values[leaves]
        Pa_residuals = a @ np.linalg.solve(a.T @ a, a.T @ residuals)
        return np.sum(np.square(residuals)) + (gamma - 1) * residuals.T @ Pa_residuals

    def probit_loss(leaf_values):
        scores = f + leaf_values[leaves]
        p = scipy.stats.norm.cdf(scores)
        dp = scipy.stats.norm.pdf(scores)
        losses = -np.log(np.where(y == 1, p, 1 - p))
        dl = np.where(y == 1, -dp / p, dp / (1 - p))
        Pa_dl = a @ np.linalg.solve(a.T @ a, a.T @ dl)
        return np.sum(losses) + (gamma - 1) / 2 * dl.T @ Pa_dl

    if objective == "regression":
        loss = regression_loss
    else:
        loss = probit_loss

    def vectorize(f):
        def f_(x):
            shape = x.shape
            x = x.reshape((shape[0], -1))
            out = np.array([f(x[:, i]) for i in range(x.shape[1])])
            out = out.reshape(shape[1:])
            return out

        return f_

    grad = scipy.optimize.approx_fprime(np.zeros(num_leaves), loss, 1e-6)
    hess = scipy.differentiate.hessian(vectorize(loss), np.zeros(num_leaves))
    expected_leaf_values = -np.linalg.solve(hess.ddf, grad) * learning_rate

    for i in range(num_leaves):
        assert np.allclose(
            model.booster.get_leaf_output(9, i),
            expected_leaf_values[i],
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("gamma", [1, 10])
@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_anchor_boosting_decreases_loss(gamma, objective):
    num_leaves = 5
    n = 1000

    x, y, a = simulate(f1, n=n, shift=0, seed=0)
    if objective == "binary":
        y = (y > 0).astype(int)

    model = AnchorBooster(
        gamma=gamma,
        num_boost_round=10,
        num_leaves=num_leaves,
        objective=objective,
    )
    model.fit(x, y, Z=a)

    def regression_loss(y, f, a):
        residuals = y - f
        Pa_residuals = a @ np.linalg.solve(a.T @ a, a.T @ residuals)
        return np.sum(np.square(residuals) + (gamma - 1) * np.square(Pa_residuals))

    def probit_loss(y, f, a):
        p = scipy.stats.norm.cdf(f)
        dp = scipy.stats.norm.pdf(f)
        losses = -np.log(np.where(y == 1, p, 1 - p))
        dl = np.where(y == 1, -dp / p, dp / (1 - p))

        Pa_dl = a @ np.linalg.solve(a.T @ a, a.T @ dl)
        return np.sum(losses) + (gamma - 1) * dl.T @ Pa_dl

    if objective == "regression":
        loss = regression_loss
    else:
        loss = probit_loss

    loss_value = np.inf
    for idx in range(10):
        f = model.predict(x, num_iteration=idx + 1, raw_score=True)
        new_loss_value = loss(y, f, a)

        if idx > 0:
            assert new_loss_value < loss_value

        loss_value = new_loss_value


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"max_depth": -1},
        {"num_leaves": 63},
        {"min_split_gain": 0.002},
        {"lambda_l2": 0.1},
    ],
)
def test_compare_anchor_boosting_to_lgbm(parameters):
    X, y, a = simulate(f1, shift=0, seed=0)

    lgbm_model = lgb.train(
        params={
            "learning_rate": 0.1,
            "objective": "regression",
            **parameters,
        },
        train_set=lgb.Dataset(X, y),
        num_boost_round=10,
    )

    anchor_booster = AnchorBooster(
        gamma=1,
        num_boost_round=10,
        objective="regression",
        learning_rate=0.1,
        **parameters,
    ).fit(X, y, Z=a)

    lgbm_pred = lgbm_model.predict(X)
    anchor_booster_pred = anchor_booster.predict(X)

    np.testing.assert_allclose(lgbm_pred, anchor_booster_pred, rtol=1e-5)


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_anchor_booster_init_score(objective):
    X, y, a = simulate(f1, shift=0, seed=0)

    if objective == "binary":
        y = (y > 0).astype(int)

    anchor_booster = AnchorBooster(
        gamma=1,
        num_boost_round=0,
        objective=objective,
        learning_rate=0.1,
    ).fit(X, y, Z=a)

    predictions = anchor_booster.predict(X)
    assert np.allclose(predictions, np.ones_like(y) * y.mean())
