import numpy as np


class RegressionMixin:
    def init_score(self, y):
        return np.tile(y.mean(), len(y))

    def grad(self, f, data):
        # Replicate LGBM behaviour
        # https://github.com/microsoft/LightGBM/blob/e9fbd19d7cbaeaea1ca54a091b160868fc\
        # 5c79ec/src/objective/regression_objective.hpp#L130-L131
        return -(data.get_label() - f)

    def hess(self, f, data):
        # Replicate LGBM behaviour
        # https://github.com/microsoft/LightGBM/blob/e9fbd19d7cbaeaea1ca54a091b160868fc\
        # 5c79ec/src/objective/regression_objective.hpp#L130-L131
        return np.ones(len(data.get_label()))

    def loss(self, f, data):
        return 0.5 * (data.get_label() - f) ** 2
