import lightgbm as lgb
import numpy as np

from anchorboost import (
    ClassificationMixin,
    LGBMMixin,
    MultiClassificationMixin,
    RegressionMixin,
)
from anchorboost.simulate import f1, simulate


def test_classification_to_lgbm():
    class ClassificationObjective(LGBMMixin, ClassificationMixin):
        pass

    X, y, a = simulate(f1, shift=0, seed=0)
    y = (y > 0).astype(int)

    loss = ClassificationObjective()
    data = lgb.Dataset(X, y, init_score=loss.init_score(y))

    lgb_model = lgb.train(
        params={"learning_rate": 0.1, "objective": "binary"},
        train_set=data,
        num_boost_round=10,
    )

    my_model = lgb.train(
        params={"learning_rate": 0.1},
        train_set=data,
        num_boost_round=10,
        fobj=loss.objective,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = loss.predictions(my_model.predict(X))

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)


def test_multi_classification_to_lgbm():
    class MultiClassificationObjective(LGBMMixin, MultiClassificationMixin):
        pass

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
        params={"learning_rate": 0.1, "num_class": 3},
        train_set=data,
        num_boost_round=10,
        fobj=loss.objective,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = loss.predictions(my_model.predict(X))

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)


def test_regression_to_lgbm():
    class RegressionObjective(LGBMMixin, RegressionMixin):
        pass

    X, y, a = simulate(f1, shift=0, seed=0)

    loss = RegressionObjective()
    data = lgb.Dataset(X, y, init_score=loss.init_score(y))

    lgb_model = lgb.train(
        params={"learning_rate": 0.1, "objective": "regression"},
        train_set=data,
        num_boost_round=10,
    )

    my_model = lgb.train(
        params={"learning_rate": 0.1},
        train_set=data,
        num_boost_round=10,
        fobj=loss.objective,
    )

    lgb_pred = lgb_model.predict(X)
    my_pred = my_model.predict(X)

    np.testing.assert_allclose(lgb_pred, my_pred, rtol=1e-5)
