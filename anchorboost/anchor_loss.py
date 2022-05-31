import numpy as np


class AnchorL2Loss:
    def __init__(self, gamma):
        self.gamma = gamma

    def init_score(self, y):
        return np.mean(y)

    def lgb_anchor_objective(self, y_pred, data):
        y = data.get_label()
        residuals = y - y_pred
        return -self.grad(residuals, data.anchor), self.hess(residuals, data.anchor)

    def lgb_anchor_score(self, y_pred, data):
        y = data.get_label()
        residuals = y - y_pred
        return "anchor", self.anchor_loss(residuals, data.anchor).mean(), False

    def lgb_l2_score(self, y_pred, data):
        y = data.get_label()
        return "l2", self.l2_loss(y - y_pred).mean(), False

    def _proj_matrix(self, a):
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)

    def _proj(self, a, b):
        # Linear projection of b onto the column space of a.
        return np.dot(a, np.linalg.lstsq(a, b, rcond=None)[0])

    def anchor_loss(self, residuals, anchor):
        return residuals ** 2 + (self.gamma - 1) * self._proj(anchor, residuals) ** 2

    def l2_loss(self, residuals):
        return residuals ** 2

    def grad(self, residuals, anchor):
        return 2.0 * residuals + (self.gamma - 1) * self._proj(anchor, residuals)

    def hess(self, residuals, anchor):
        return 2 * np.ones(residuals.shape[0])
