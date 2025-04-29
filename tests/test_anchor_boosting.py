import numpy as np
import pytest

from anchorboosting import AnchorBooster
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_anchor_boosting(gamma):
    learning_rate = 0.1
    num_leaves = 3
    n = 1000

    x, y, a = simulate(f1, n=n, shift=0, seed=0)
    # Fit the model
    model = AnchorBooster(gamma=gamma, num_boost_round=10, num_leaves=num_leaves)
    model.fit(x, y, Z=a)

    residuals = y - model.predict(x, num_iteration=9)
    residuals += (np.sqrt(gamma) - 1) * a @ np.linalg.pinv(a) @ residuals

    leaves = model.booster.predict(
        x, pred_leaf=True, start_iteration=9, num_iteration=1
    )
    M = np.equal.outer(leaves, np.arange(num_leaves)).astype(float)
    M += (np.sqrt(gamma) - 1) * a @ np.linalg.pinv(a) @ M

    expected_leaf_values = np.linalg.lstsq(M, residuals, rcond=None)[0] * learning_rate
    for i in range(num_leaves):
        assert np.allclose(
            model.booster.get_leaf_output(9, i),
            expected_leaf_values[i],
            atol=1e-5,
            rtol=1e-5,
        )
