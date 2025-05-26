import numpy as np
import pytest

from anchorboosting.models import AnchorBooster
from anchorboosting.simulate import f1, simulate


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_compare_refit_to_lgbm(objective):

    X, y, a = simulate(f1, shift=0, seed=0)

    if objective == "binary":
        y = (y > 0).astype(int)

    anchor_booster = AnchorBooster(
        objective=objective,
        gamma=1,
        num_boost_round=10,
    )

    anchor_booster.fit(X, y, Z=a)
    yhat = anchor_booster.predict(X)

    new_anchor_booster = anchor_booster.refit(X[:20], y[:20], decay_rate=0.5)

    # Make sure .refit does not change the model, but returns a copy
    np.testing.assert_allclose(yhat, anchor_booster.predict(X), rtol=1e-5)

    # .refit changes the original model
    assert not np.allclose(yhat, new_anchor_booster.predict(X), rtol=1e-5)

    # refitting on the same data should not change the model
    new_anchor_booster = anchor_booster.refit(X, y, decay_rate=0.5)
    np.testing.assert_allclose(yhat, new_anchor_booster.predict(X), rtol=1e-5)
