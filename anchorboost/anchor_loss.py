import numpy as np


class AnchorL2Loss:
    def __init__(self, gamma, anchor_variables):
        self.gamma = gamma
        self.anchor_variables = anchor_variables

    def init_score(self, y):
        return np.mean(y)

    def lgb_objective(self, y_pred, data):
        y = data.get_label()
        residuals = y - y_pred
        anchor = data.get_data()[:, self.anchor_variables]

        return self.grad(residuals, anchor), self.hess(residuals, anchor)

    def lgb_score(self, y_pred, data):
        y = data.get_label()
        residuals = y - y_pred
        anchor = data.get_data()[:, self.anchor_variables]

        return "anchor_l2", self.loss(residuals, anchor).mean(), False

    def _proj_matrix(self, a):
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)

    def _proj(self, a, b):
        return np.dot(a, np.linalg.lstsq(a, b)[0])

    def loss(self, residuals, anchor):
        return residuals ** 2 + (self.gamma - 1) * self._proj(anchor, residuals) ** 2

    def grad(self, residuals, anchor):
        return 2.0 * residuals + 0 * (self.gamma - 1) * self._proj(anchor, residuals)

    def hess(self, _, anchor):
        return 2 * (
            np.eye(anchor.shape[0]) + (self.gamma - 1) * self._proj_matrix(anchor)
        )
