import numpy as np
import pytest
import scipy

from anchorboosting.models import AnchorBooster
from anchorboosting.simulate import f1, f2, simulate


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("fun", [f1, f2])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_refit(objective, seed, fun):

    X, y, a = simulate(fun, shift=0, seed=seed)

    if objective == "binary":
        y = (y > 0).astype(int)

    anchor_booster = AnchorBooster(
        objective=objective,
        gamma=1,
        num_boost_round=10,
    )

    anchor_booster.fit(X, y, Z=a)
    yhat = anchor_booster.predict(X, raw_score=True)

    same_anchor_booster = anchor_booster.refit(X[:20], y[:20], decay_rate=1)
    same_yhat = same_anchor_booster.predict(X, raw_score=True)

    np.testing.assert_array_equal(yhat, same_yhat)

    same_anchor_booster = anchor_booster.refit(X, y, decay_rate=0.5)
    same_yhat = same_anchor_booster.predict(X, raw_score=True)

    np.testing.assert_array_equal(yhat, same_yhat)

    new_anchor_booster = anchor_booster.refit(X[:20], y[:20], decay_rate=0)
    new_yhat = new_anchor_booster.predict(X[:20], raw_score=True)

    def regression_loss(f, y):
        return np.sum(np.square(y - f))

    def probit_loss(f, y):
        p = scipy.stats.norm.cdf(f)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    if objective == "regression":
        loss = regression_loss
    else:
        loss = probit_loss

    assert loss(yhat[:20], y[:20]) > loss(new_yhat, y[:20])
