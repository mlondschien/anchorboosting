import lightgbm as lgb
import numpy as np
import pytest

from anchorboosting.objectives.anchor_objectives import (
    AnchorKookClassificationObjective,
    AnchorRegressionObjective,
)
from anchorboosting.objectives.mixins import (
    ClassificationMixin,
    LGBMMixin,
    MultiClassificationMixin,
    RegressionMixin,
)
from anchorboosting.simulate import f1, simulate


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
    loss2 = AnchorKookClassificationObjective(gamma=1)

    data0 = lgb.Dataset(X, y)
    data1 = lgb.Dataset(X, y, init_score=loss1.init_score(y))
    data2 = lgb.Dataset(X, y, init_score=loss2.init_score(y))
    data2.anchor = a

    params = {
        "random_state": 0,
        "deterministic": True,
        "verbosity": -1,
        "learning_rate": 0.1,
    }
    model0 = lgb.train(
        params={**params, "objective": "binary", **parameters},
        train_set=data0,
        num_boost_round=10,
    )
    model1 = lgb.train(
        params={**params, "objective": loss1.objective, **parameters},
        train_set=data1,
        num_boost_round=10,
    )

    model2 = lgb.train(
        params={**params, "objective": loss2.objective, **parameters},
        train_set=data2,
        num_boost_round=10,
    )

    pred0 = model0.predict(X)
    pred1 = loss1.predictions(model1.predict(X) + loss1.init_score(y))
    pred2 = loss2.predictions(model2.predict(X) + loss2.init_score(y))

    np.testing.assert_allclose(pred0, pred1, rtol=1e-5)
    np.testing.assert_allclose(pred0, pred2, rtol=1e-5)


def test_multi_classification_to_lgbm():
    X, y, _ = simulate(f1, shift=0, seed=0)
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
        # {"colsample_bytree": 0.6}, https://github.com/microsoft/LightGBM/issues/5543
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

    data0 = lgb.Dataset(X, y, init_score=np.ones_like(y) * np.mean(y))
    data1 = lgb.Dataset(X, y, init_score=loss1.init_score(y))
    data2 = lgb.Dataset(X, y, init_score=loss2.init_score(y))
    data2.anchor = a

    model0 = lgb.train(
        params={"learning_rate": 0.1, "objective": "regression", **parameters},
        train_set=data0,
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

    pred0 = model0.predict(X) + y.mean()
    pred1 = model1.predict(X) + loss1.init_score(y)
    pred2 = model2.predict(X) + loss2.init_score(y)

    np.testing.assert_allclose(pred0, pred1, rtol=1e-6)
    np.testing.assert_allclose(pred0, pred2, rtol=1e-6)


@pytest.mark.parametrize(
    "decay_rate",
    [0.0, 0.5, 1.0],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {},
        # {"colsample_bytree": 0.6}, https://github.com/microsoft/LightGBM/issues/5543
        {"max_depth": -1},
        {"num_leaves": 63},
        {"min_split_gain": 0.002},
        {"reg_lambda": 0.1},
        {"subsample_freq": 0},
        {"subsample": 0.2},
    ],
)
def test_compare_refit_to_lgbm(parameters, decay_rate):
    class RegressionObjective(LGBMMixin, RegressionMixin):
        pass

    X, y, a = simulate(f1, shift=0, seed=0)

    loss1 = RegressionObjective()
    loss2 = AnchorRegressionObjective(gamma=1)

    data0 = lgb.Dataset(X, y, init_score=loss1.init_score(y))
    data1 = lgb.Dataset(X, y, init_score=loss2.init_score(y))
    data2 = lgb.Dataset(X, y, init_score=loss2.init_score(y))
    data2.anchor = a

    model0 = lgb.train(
        params={"learning_rate": 0.1, "objective": "regression", **parameters},
        train_set=data0,
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

    X_new, y_new, _ = simulate(f1, shift=1, seed=0)
    model0 = model0.refit(
        data=X_new,
        label=y_new,
        decay_rate=decay_rate,
        init_score=loss1.init_score(y_new),
    )
    model1._Booster__set_objective_to_none = False
    model1.params["objective"] = "regression"
    model1 = model1.refit(
        data=X_new,
        label=y_new,
        decay_rate=decay_rate,
        init_score=loss1.init_score(y_new),
    )
    model2._Booster__set_objective_to_none = False
    model2.params["objective"] = "regression"
    model2 = model2.refit(
        data=X_new,
        label=y_new,
        decay_rate=decay_rate,
        init_score=loss2.init_score(y_new),
    )

    pred0 = model0.predict(X_new)
    pred1 = model1.predict(X_new)
    pred2 = model2.predict(X_new)
    np.testing.assert_allclose(pred0, pred1, rtol=1e-5)
    np.testing.assert_allclose(pred0, pred2, rtol=1e-5)
