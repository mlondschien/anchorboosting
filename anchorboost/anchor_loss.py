import numpy as np
from scipy.special import logsumexp


class AnchorMixin:

    name = "mixin"
    higher_is_better = False

    def __init__(self, gamma):
        self.gamma = gamma

    def loss(self, f, y, anchor):
        residuals = self.residuals(f, y)

        if self.gamma == 1:
            return residuals ** 2

        return residuals ** 2 + (self.gamma - 1) * self._proj(anchor, residuals) ** 2

    def objective(self, f, data):
        """Objective function for LGBM."""
        y = data.get_label()
        anchor = data.anchor
        return self.grad(f, y, anchor), self.hess(f, y, anchor)

    def score(self, f, data):
        """Score function for LGBM."""
        y = data.get_label()
        anchor = data.anchor
        return (
            f"{self.name}_{self.gamma}",
            self.loss(f, y, anchor).mean(),
            self.higher_is_better,
        )

    def _proj(self, anchor, f):
        """Project f onto the subspace spanned by anchor.

        Parameters
        ----------
        anchor: np.ndarray of dimension (n, d_anchor).
            The anchor matrix. If d_anchor = 1 and entries are integer, assumed to be
            categories.
        f: np.ndarray of dimension (n, d_f).
            The vector to project.

        Returns
        -------
        np.ndarray of dimension (n, d_f).
            Projection of f onto the subspace spanned by anchor.
        """
        # If anchor is categorical (i.e., corresponds to environments), the projection
        # onto the anchor is just the environment-wise mean.
        if anchor.shape[1] == 1 and "int" in str(anchor.dtype):
            projected_values = np.zeros(f.shape)
            for unique_value in np.unique(anchor):
                mask = anchor.flatten() == unique_value
                projected_values[mask, :] = f[mask, :].mean(axis=0)
            return projected_values

        return np.dot(anchor, np.linalg.lstsq(anchor, f, rcond=None)[0])

    def _proj_matrix(self, a):
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)

    def hess(self, f, y, anchor):
        """Trivial hessian."""
        return 2 * np.ones(f.size)


class AnchorRegressionLoss(AnchorMixin):

    name = "anchor_regression"

    def init_score(self, y):
        return np.mean(y)

    def residuals(self, f, y):
        return y - f

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
        if self.gamma == 1:
            return self.negative_log_likelihood(f, y)

        residuals = self.residuals(f, y)
        loss = self.negative_log_likelihood(f, y) + (self.gamma - 1) * np.sum(
            self._proj(anchor, residuals) ** 2, axis=1
        )
        return loss

    def predictions(self, f):
        f = f - np.max(f)  # normalize f to avoid overflow
        divisor = np.sum(np.exp(f), axis=1)
        predictions = np.exp(f) / divisor[:, np.newaxis]

        return predictions

    def residuals(self, f, y):
        f = f - np.max(f)  # normalize f to avoid overflow
        divisor = np.sum(np.exp(f), axis=1)
        residuals = np.exp(f) / divisor[:, np.newaxis]
        indices = self._indices(y)
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

    def _indices(self, y):
        return (np.arange(len(y)), y.astype(int))

    def negative_log_likelihood(self, f, y):
        f = f - np.max(f)
        indices = self._indices(y)
        log_divisor = logsumexp(f, axis=1)[:, np.newaxis]
        return -f[indices] + log_divisor
