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

    x, y, a = simulate(f1, n=n, shift=0, seed=0)

    if objective == "binary":
        y = (y > 0).astype(int)

    model = AnchorBooster(
        gamma=gamma,
        num_boost_round=10,
        num_leaves=num_leaves,
        objective=objective,
        learning_rate=learning_rate,
    )
    model.fit(x, y, Z=a)

    f = model.predict(x, num_iteration=9, raw_score=True)

    leaves = model.booster.predict(
        x, pred_leaf=True, start_iteration=9, num_iteration=1
    ).flatten()

    def regression_loss(leaf_values):
        residuals = y - f - leaf_values[leaves]
        Pa_residuals = a @ np.linalg.solve(a.T @ a, a.T @ residuals)
        return np.sum(np.square(residuals)) + (gamma - 1) * residuals.T @ Pa_residuals

    def classification_loss(leaf_values):
        scores = f + leaf_values[leaves]
        p = 1 / (1 + np.exp(-scores))
        residuals = y - p
        Pa_residuals = a @ np.linalg.solve(a.T @ a, a.T @ residuals)
        return (
            np.sum(-np.log(np.where(y == 1, p, 1 - p)))
            + (gamma - 1) * residuals.T @ Pa_residuals
        )

    if objective == "regression":
        loss = regression_loss
    else:
        loss = classification_loss

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


@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_anchor_boosting_decreases_loss(gamma, objective):
    # learning_rate = 0.1
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
        honest_splits=False,
    )
    model.fit(x, y, Z=a)

    def regression_loss(y, f, a):
        residuals = y - f
        Pa_residuals = a @ np.linalg.inv(a.T @ a) @ a.T @ residuals
        return np.sum(np.square(residuals) + (gamma - 1) * np.square(Pa_residuals))

    def classification_loss(y, f, a):
        p = 1 / (1 + np.exp(-f))
        residuals = y - p
        Pa_residuals = a @ np.linalg.inv(a.T @ a) @ a.T @ residuals
        return np.mean(
            -np.log(np.where(y == 1, p, 1 - p)) + (gamma - 1) * np.square(Pa_residuals)
        )

    loss = regression_loss if objective == "regression" else classification_loss

    loss_value = np.inf
    for idx in range(10):
        f = model.predict(x, num_iteration=idx + 1, raw_score=True)
        new_loss_value = loss(y, f, a)

        if idx > 0:
            assert new_loss_value < loss_value

        loss_value = new_loss_value


# @pytest.mark.parametrize("gamma", [1.0, 2.0, 100])
# def test_anchor_boosting(gamma):
#     learning_rate = 0.1
#     num_leaves = 3
#     n = 1000

#     rng = np.random.default_rng(0)
#     mask = np.ones(n, dtype=bool)
#     mask[500:] = False
#     for _ in range(10):
#         rng.shuffle(mask)

#     x, y, a = simulate(f1, n=n, shift=0, seed=0)

#     model = AnchorBooster(gamma=gamma, num_boost_round=10, num_leaves=num_leaves)
#     model.fit(x, y, Z=a)

#     x, y = x[~mask, :], y[~mask]
#     residuals = y - model.predict(x, num_iteration=9)

#     leaves = model.booster.predict(
#         x, pred_leaf=True, start_iteration=9, num_iteration=1
#     ).flatten()

#     # In Anchor Regression, we optimize || (Id + (gamma - 1) P_Z) (y - M beta) ||^2
#     # In honest splits, we want to use y[mask] only to estimate split parameters, not
#     # the leaf values. However, we still use all of the anchor `a` to estimate its
#     # covariance `a.T @ a`.
#     M = np.equal.outer(leaves, np.arange(num_leaves)).astype(float)
#     MTM = M.T @ M + (gamma - 1) * M.T @ a[~mask, :] @ np.linalg.solve(
#         a.T @ a, a[~mask, :].T @ M
#     )
#     MTy = M.T @ residuals + (gamma - 1) * M.T @ a[~mask, :] @ np.linalg.solve(
#         a.T @ a, a[~mask, :].T @ residuals
#     )

#     expected_leaf_values = np.linalg.solve(MTM, MTy) * learning_rate
#     for i in range(num_leaves):
#         assert np.allclose(
#             model.booster.get_leaf_output(9, i),
#             expected_leaf_values[i],
