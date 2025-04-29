import numpy as np
import pytest

from anchorboosting import AnchorBooster
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_anchor_boosting(gamma):
    learning_rate = 0.1
    num_leaves = 3
    n = 1000

    rng = np.random.default_rng(0)
    mask = np.ones(n, dtype=bool)
    mask[500:] = False
    for _ in range(10):
        rng.shuffle(mask)

    x, y, a = simulate(f1, n=n, shift=0, seed=0)

    model = AnchorBooster(gamma=gamma, num_boost_round=10, num_leaves=num_leaves)
    model.fit(x, y, Z=a)

    x, y = x[~mask, :], y[~mask]
    residuals = y - model.predict(x, num_iteration=9)

    leaves = model.booster.predict(
        x, pred_leaf=True, start_iteration=9, num_iteration=1
    ).flatten()

    # In Anchor Regression, we optimize || (Id + (gamma - 1) P_Z) (y - M beta) ||^2
    # In honest splits, we want to use y[mask] only to estimate split parameters, not
    # the leaf values. However, we still use all of the anchor `a` to estimate its
    # covariance `a.T @ a`.
    M = np.equal.outer(leaves, np.arange(num_leaves)).astype(float)
    MTM = M.T @ M + (gamma - 1) * M.T @ a[~mask, :] @ np.linalg.solve(
        a.T @ a, a[~mask, :].T @ M
    )
    MTy = M.T @ residuals + (gamma - 1) * M.T @ a[~mask, :] @ np.linalg.solve(
        a.T @ a, a[~mask, :].T @ residuals
    )

    expected_leaf_values = np.linalg.solve(MTM, MTy) * learning_rate
    for i in range(num_leaves):
        assert np.allclose(
            model.booster.get_leaf_output(9, i),
            expected_leaf_values[i],
            atol=1e-5,
            rtol=1e-5,
        )
