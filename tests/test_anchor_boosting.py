import lightgbm as lgb
import numpy as np
import pytest
import scipy

from anchorboosting.models import AnchorBooster, Proj
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("gamma", [1.0, 2.0, 100])
@pytest.mark.parametrize("objective", ["regression", "binary"])
@pytest.mark.parametrize("categorical_z", [True, False])
def test_anchor_boosting_second_order(gamma, objective, categorical_z):
    learning_rate = 0.1
    num_leaves = 5
    n = 200
    num_boost_round = 10

    x, y, a = simulate(f1, n=n, shift=0, seed=0)
    if categorical_z:
        a = np.digitize(a[:, 0], bins=[-1, 0, 1, 2, 3])
        assert np.issubdtype(a.dtype, np.integer)

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

    leaves = model.booster_.predict(
        x.to_arrow(),
        pred_leaf=True,
        start_iteration=num_boost_round - 1,
        num_iteration=1,
    ).flatten()

    proj = Proj(a)

    def regression_loss(leaf_values):
        residuals = y - f - leaf_values[leaves]
        Pa_residuals = proj(residuals)
        return np.sum(np.square(residuals)) + (gamma - 1) * residuals.T @ Pa_residuals

    def probit_loss(leaf_values):
        scores = f + leaf_values[leaves]
        p = scipy.stats.norm.cdf(scores)
        dp = scipy.stats.norm.pdf(scores)
        losses = -np.log(np.where(y == 1, p, 1 - p))
        dl = np.where(y == 1, -dp / p, dp / (1 - p))
        Pa_dl = proj(dl)
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
            model.booster_.get_leaf_output(9, i),
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
        train_set=lgb.Dataset(X.to_arrow(), y, categorical_feature=[2]),
        num_boost_round=50,
    )

    anchor_booster = AnchorBooster(
        gamma=1,
        num_boost_round=50,
        objective="regression",
        learning_rate=0.1,
        **parameters,
    ).fit(X, y, Z=a, categorical_feature=["x3"])

    anchor_booster_noncat = AnchorBooster(
        gamma=1,
        num_boost_round=50,
        objective="regression",
        learning_rate=0.1,
        **parameters,
    ).fit(X, y, Z=a)

    lgbm_pred = lgbm_model.predict(X.to_arrow())
    anchor_booster_pred = anchor_booster.predict(X)
    anchor_booster_noncat_pred = anchor_booster_noncat.predict(X)

    np.testing.assert_allclose(lgbm_pred, anchor_booster_pred, rtol=1e-5)
    assert not np.allclose(lgbm_pred, anchor_booster_noncat_pred)


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_categorical_and_onehot_is_the_same(objective):
    X, y, a = simulate(f1, shift=0, seed=0)

    a = np.digitize(a[:, 0], bins=[-1, 0, 1, 2, 3])

    if objective == "binary":
        y = (y > 0).astype(int)

    anchor_booster1 = AnchorBooster(
        gamma=1,
        num_boost_round=10,
        objective=objective,
        learning_rate=0.1,
    ).fit(X, y, Z=a)

    predictions1 = anchor_booster1.predict(X)

    M = scipy.sparse.csr_matrix(
        (np.ones(len(a), dtype=np.float64), (np.arange(len(a)), a)),
        shape=(len(a), len(np.unique(a))),
    )
    anchor_booster2 = AnchorBooster(
        gamma=1,
        num_boost_round=10,
        objective=objective,
        learning_rate=0.1,
    ).fit(X, y, Z=M.toarray())
    predictions2 = anchor_booster2.predict(X)
    np.testing.assert_allclose(predictions1, predictions2, rtol=1e-5)
