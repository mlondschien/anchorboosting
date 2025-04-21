import lightgbm as lgb
import numpy as np
import pytest

from anchorboosting import AnchorLiuClassificationObjective


@pytest.mark.parametrize("y", [0, 1])
@pytest.mark.parametrize("f", [-2, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 2])
def test_liu_residuals(y, f):
    loss = AnchorLiuClassificationObjective(gamma=1)
    data = lgb.Dataset(None, np.array([y]))
    residual = loss.residuals(np.array([f]), data)[0]

    n_sample = 100000

    y = 2 * y - 1
    truncated_logistic = draw_truncated_logistic(f, y, n_sample) - f
    assert np.abs(np.mean(truncated_logistic) - residual) < 1e-2


def draw_truncated_logistic(mu, y, size):
    out = np.zeros(size)
    mask = np.ones(size, dtype=bool)
    rng = np.random.RandomState(0)
    while mask.any():
        out[mask] = rng.logistic(mu, 1, mask.sum())
        mask = y * out < 0

    return out
