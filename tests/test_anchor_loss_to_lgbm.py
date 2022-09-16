import lightgbm as lgb
import numpy as np

from anchorboost.anchor_loss import AnchorClassificationLoss, AnchorRegressionLoss
from anchorboost.simulate import f1, simulate


def test_regression():
    l2_loss = AnchorRegressionLoss(gamma=1)

    X, y, a = simulate(f1, shift=0, seed=0)
    data = lgb.Dataset(X, y, init_score=l2_loss.init_score(y))
    data.anchor = a
    lgb_model = lgb.train(
        params={"learning_rate": 0.1},
        train_set=data,
        num_boost_round=10,
    )

    my_model = lgb.train(
        params={"learning_rate": 0.1},
        train_set=data,
        num_boost_round=10,
        fobj=l2_loss.objective,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = my_model.predict(X)

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)


def test_classification():
    X, y, a = simulate(f1, shift=0, seed=0)
    y = (y > 0).astype(int) + (y > 1).astype(int)

    loss = AnchorClassificationLoss(gamma=1, n_classes=3)
    data = lgb.Dataset(X, y, init_score=loss.init_score(y))
    data.anchor = a

    lgb_model = lgb.train(
        params={"learning_rate": 0.1, "objective": "multiclass", "num_class": 3},
        train_set=data,
        num_boost_round=10,
    )

    my_model = lgb.train(
        params={"learning_rate": 0.1, "num_class": 3},
        train_set=data,
        num_boost_round=10,
        fobj=loss.objective,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = loss.predictions(my_model.predict(X))

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)
