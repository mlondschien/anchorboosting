import numpy as np
import pytest

from anchorboosting.models import AnchorBooster
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("gamma", [1.0, 2.0, 100])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_compare_refit_to_lgbm(objective, gamma):

    X, y, a = simulate(f1, shift=0, seed=0)

    if objective == "binary":
        y = (y > 0).astype(int)

    anchor_booster = AnchorBooster(
        objective=objective,
        gamma=gamma,
        num_boost_round=10,
    )

    anchor_booster.fit(X, y, Z=a)
    anchor_booster.update(X, y, Z=a, num_iteration=10)
    yhat = anchor_booster.predict(X)

    new_anchor_booster = AnchorBooster(
        objective=objective,
        gamma=gamma,
        num_boost_round=20,
    )
    new_anchor_booster.fit(X, y, Z=a)
    new_yhat = new_anchor_booster.predict(X)

    np.testing.assert_allclose(yhat, new_yhat, rtol=1e-5)
