import lightgbm as lgb
import numpy as np
import pytest

from anchorboost import (
    AnchorKookClassificationObjective,
    AnchorRegressionObjective,
    ClassificationMixin,
    LGBMMixin,
    MultiClassificationMixin,
    RegressionMixin,
)
from anchorboost.simulate import f1, simulate


class ClassificationObjective(LGBMMixin, ClassificationMixin):
    pass


class MultiClassificationObjective(LGBMMixin, MultiClassificationMixin):
    pass


@pytest.mark.parametrize(
    "parameters",
    [
        # {"colsample_bytree": 0.6},
        {"max_depth": -1},
        {"num_leaves": 63},
        {"learning_rate": 0.025},
        {"min_split_gain": 0.002},
        {"reg_lambda": 0.1},
        {"subsample_freq": 0},
        {"subsample": 0.2},
    ],
)
def test_classification_to_lgbm(parameters):
    X, y, a = simulate(f1, shift=0, seed=0)
    y = (y > 0).astype(int)

    loss1 = ClassificationObjective()
    loss2 = MultiClassificationObjective(2)
    loss3 = AnchorKookClassificationObjective(gamma=1)

    data1 = lgb.Dataset(X, y, init_score=loss1.init_score(y))
    data2 = lgb.Dataset(X, y, init_score=loss2.init_score(y))
    data3 = lgb.Dataset(X, y, init_score=loss3.init_score(y))
    data3.anchor = a

    params = {
        "random_state": 0,
        "deterministic": True,
        "verbosity": -1,
        "learning_rate": 0.1,
    }
    model0 = lgb.train(
        params={**params, "objective": "binary", **parameters},
        train_set=data1,
        num_boost_round=10,
    )
    model1 = lgb.train(
        params={**params, "objective": loss1.objective, **parameters},
        train_set=data1,
        num_boost_round=10,
    )

    model2 = lgb.train(
        params={**params, "objective": loss2.objective, "num_class": 2, **parameters},
        train_set=data2,
        num_boost_round=10,
    )

    model3 = lgb.train(
        params={**params, "objective": loss3.objective, **parameters},
        train_set=data3,
        num_boost_round=10,
    )

    pred0 = model0.predict(X)
    pred1 = loss1.predictions(model1.predict(X))
    pred2 = loss2.predictions(model2.predict(X))[:, 1]
    pred3 = loss3.predictions(model3.predict(X))

    np.testing.assert_allclose(pred0, pred1, rtol=1e-5)

    np.testing.assert_allclose(pred0, pred3, rtol=1e-5)

    if "reg_lambda" not in parameters:
        np.testing.assert_allclose(pred0, pred2, rtol=1e-5)


def test_multi_classification_to_lgbm():

    X, y, a = simulate(f1, shift=0, seed=0)
    y = (y > 0).astype(int) + (y > 1).astype(int)

    loss = MultiClassificationObjective(n_classes=3)
    data = lgb.Dataset(X, y, init_score=loss.init_score(y))

    lgb_model = lgb.train(
        params={"learning_rate": 0.1, "objective": "multiclass", "num_class": 3},
        train_set=data,
        num_boost_round=10,
    )

    my_model = lgb.train(
        params={"learning_rate": 0.1, "num_class": 3, "objective": loss.objective},
        train_set=data,
        num_boost_round=10,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = loss.predictions(my_model.predict(X))

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        # {"colsample_bytree": 0.6},
        {"max_depth": -1},
        {"num_leaves": 63},
        {"min_split_gain": 0.002},
        {"reg_lambda": 0.1},
        {"subsample_freq": 0},
        {"subsample": 0.2},
    ],
)
def test_regression_to_lgbm(parameters):
    class RegressionObjective(LGBMMixin, RegressionMixin):
        pass

    X, y, a = simulate(f1, shift=0, seed=0)

    loss1 = RegressionObjective()
    loss2 = AnchorRegressionObjective(gamma=1)

    data1 = lgb.Dataset(X, y, init_score=loss1.init_score(y))
    data2 = lgb.Dataset(X, y, init_score=loss2.init_score(y))

    model0 = lgb.train(
        params={"learning_rate": 0.1, "objective": "regression", **parameters},
        train_set=data1,
        num_boost_round=10,
    )

    model1 = lgb.train(
        params={"learning_rate": 0.1, "objective": loss1.objective, **parameters},
        train_set=data1,
        num_boost_round=10,
    )

    model2 = lgb.train(
        params={"learning_rate": 0.1, "objective": loss2.objective, **parameters},
        train_set=data2,
        num_boost_round=10,
    )

    pred0 = model0.predict(X)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    np.testing.assert_allclose(pred0, pred1, rtol=1e-5)
    np.testing.assert_allclose(pred0, pred2, rtol=1e-5)
