import numpy as np


class AnchorMixin:

    name = "mixin"
    higher_is_better = False

    def __init__(self, gamma):
        self.gamma = gamma

    def loss(self, f, y, anchor):
        residuals = self.residuals(f, y)
        return residuals ** 2 + (self.gamma - 1) * self._proj(anchor, residuals) ** 2

    def objective(self, f, data):
        y = data.get_label()
        anchor = data.anchor
        return self.grad(f, y, anchor), self.hess(f, y, anchor)

    def score(self, f, data):
        y = data.get_label()
        anchor = data.anchor
        return self.name, self.loss(f, y, anchor).mean(), self.higher_is_better

    def _proj(self, anchor, f):
        # Linear projection of f onto the column space of anchor.
        return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0])
        # return np.dot(self._proj_matrix(anchor), f)

    def _proj_matrix(self, a):
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)

    def hess(self, f, y, anchor):
        return 2 * np.ones(f.size)


class AnchorRegressionLoss(AnchorMixin):

    name = "anchor_regression"

    def init_score(self, y):
        return np.mean(y)

    def residuals(self, f, y):
        return y - f

    def l2_loss(self, f, y):
        residuals = self.residuals(f, y)
        return residuals ** 2

    def grad(self, f, y, anchor):
        residuals = self.residuals(f, y)
        return -2.0 * residuals + -2.0 * (self.gamma - 1) * self._proj(
            anchor, residuals
        )

    def hess(self, f, y, anchor):
        return 2.0 * np.ones(len(y))


class AnchorClassificationLoss(AnchorMixin):

    name = "anchor_classification"

    def init_score(self, y):
        unique_values, unique_counts = np.unique(y, return_counts=True)
        assert (sorted(unique_values) == unique_values).all()
        return np.array([counts / sum(unique_counts) for counts in unique_counts])

    def objective(self, f, data):
        y = data.get_label()
        f = np.reshape(f, (len(y), -1))
        anchor = data.anchor
        return self.grad(f, y, anchor).flatten("C"), self.hess(f, y, anchor)

    def score(self, f, data):
        y = data.get_label()
        f = np.reshape(f, (len(y), -1))
        anchor = data.anchor
        return self.name, self.loss(f, y, anchor).mean(), True

    def loss(self, f, y, anchor):
        residuals = self.residuals(f, y)
        return self.negative_log_likelihood(f, y) + (self.gamma - 1) * np.sum(
            self._proj(anchor, residuals) ** 2, axis=1
        )

    def predictions(self, f):
        f = f - np.max(f)  # normalize f to avoid overflow
        divisor = np.sum(np.exp(f), axis=1)
        predictions = np.exp(f) / divisor[:, np.newaxis]

        return predictions

    def residuals(self, f, y):
        f = f - np.max(f)  # normalize f to avoid overflow
        divisor = np.sum(np.exp(f), axis=1)
        residuals = np.exp(f) / divisor[:, np.newaxis]
        indices = self._indices(y, f.shape[1])
        residuals[indices] -= 1
        return residuals

    def grad(self, f, y, anchor):
        f = f - np.max(f, axis=1)[:, np.newaxis]  # normalize f to avoid overflow
        divisor = np.sum(np.exp(f), axis=1)
        predictions = np.exp(f) / divisor[:, np.newaxis]
        residuals = self.residuals(f, y)
        projected_residuals = self._proj(anchor, residuals)
        grad = predictions * (
            -np.sum(predictions * projected_residuals, axis=1)[:, np.newaxis]
            + projected_residuals
        )
        return +2 * residuals + 2 * (self.gamma - 1) * grad

    def _indices(self, y, n_classes):
        return (np.arange(len(y)), y.astype(int))

    def negative_log_likelihood(self, f, y):
        f = f - np.max(f)
        indices = self._indices(y, f.shape[1])
        log_divisor = np.log(np.sum(np.exp(f), axis=1))[:, np.newaxis]
        return -f[indices] + log_divisor
