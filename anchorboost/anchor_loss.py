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
            return residuals**2

        return residuals**2 + (self.gamma - 1) * self._proj(anchor, residuals) ** 2

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
        """Projection matrix onto the subspace spanned by a."""
        assert a.shape[1] < a.shape[0]
        return np.dot(np.dot(a, np.linalg.inv(a.T @ a)), a.T)

    def hess(self, f, y, anchor):
        """Trivial hessian for LGBM."""
        return 2 * np.ones(f.size)


class AnchorRegressionLoss(AnchorMixin):

    name = "anchor_regression"

    def init_score(self, y):
        return np.tile(np.mean(y), len(y))

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

    def __init__(self, gamma, n_classes):
        super().__init__(gamma)
        self.n_classes = n_classes

    def init_score(self, y):
        """Initial score for LGBM.

        Parameters
        ----------
        y: np.ndarray of dimension (n,).
            Vector with true labels.

        Returns
        -------
        np.ndarray of length n * n_classes.
            Initial scores for LGBM. Note that this is flattened.
        """
        unique_values, unique_counts = np.unique(y, return_counts=True)
        assert len(unique_values) == self.n_classes
        assert (sorted(unique_values) == unique_values).all()
        return np.tile(
            np.array(unique_counts) / np.sum(unique_counts), (len(y), 1)
        ).flatten("C")

    def objective(self, f, data):
        """Objective function for LGBM.

        Note that LGBM will supply a 1D array for f.

        Parameters
        ----------
        f: np.ndarray of length n * n_classes.
            The vector to project.
        data: lgb.Dataset
            The dataset.

        Returns
        -------
        np.ndarray of length n * n_classes
            Gradient of the loss, flattened.
        np.ndarray of length n * n_classes
            Trivial hessian.
        """
        y = data.get_label()
        f = np.reshape(f, (len(y), self.n_classes))
        anchor = data.anchor
        return self.grad(f, y, anchor).flatten("C"), self.hess(f, y, anchor)

    def score(self, f, data):
        """Score function for LGBM.

        Note that LGBM will supply a 1D array for f.

        Parameters
        ----------
        f: np.ndarray of length n * n_classes.
            Vector with predictions.
        data: lgb.Dataset
            The dataset.

        Returns
        -------
        str
            Name of the score.
        float
            Value of the score.
        bool
            Whether higher is better.
        """
        y = data.get_label()
        f = np.reshape(f, (len(y), self.n_classes))
        anchor = data.anchor
        return self.name, self.loss(f, y, anchor).mean(), True

    def loss(self, f, y, anchor):
        """Loss function.

        Parameters
        ----------
        f: np.ndarray of dimension (n, n_classes).
            Vector with likelihoods.
        y: np.ndarray of dimension (n,).
            Vector with true labels.
        anchor: np.ndarray of dimension (n, d_anchor).
            The anchor matrix. If d_anchor = 1 and entries are integer, assumed to be
            categories.

        Returns
        -------
        np.ndarray of dimension (n,).
            Loss.
        """
        if self.gamma == 1:
            return self.negative_log_likelihood(f, y)

        residuals = self.residuals(f, y)
        loss = self.negative_log_likelihood(f, y) + (self.gamma - 1) * np.sum(
            self._proj(anchor, residuals) ** 2, axis=1
        )
        return loss

    def predictions(self, f):
        """Compute probability predictions from likelihoods via softmax.

        Parameters
        ----------
        f: np.ndarray of dimension (n, n_classes).
            Vector with likelihoods.

        Returns
        -------
        np.ndarray of dimension (n, n_classes).
            Vector with probabilities.
        """
        assert len(f.shape) == 2 and f.shape[1] == self.n_classes

        f = f - np.max(f)  # normalize f to avoid overflow
        predictions = np.exp(f)
        predictions /= np.sum(predictions, axis=1, keepdims=True)
        return predictions

    def residuals(self, f, y):
        """Compute residuals from likelihoods and true labels.

        Parameters
        ----------
        f: np.ndarray of dimension (n, n_classes).
            Vector with likelihoods.
        y: np.ndarray of dimension (n,).
            Vector with true labels.

        Returns
        -------
        np.ndarray of dimension (n, n_classes).
            Vector with residuals.
        """
        assert f.shape == (len(y), self.n_classes)

        residuals = self.predictions(f)
        residuals[self._indices(y)] -= 1
        return residuals

    def grad(self, f, y, anchor):
        assert f.shape == (len(y), self.n_classes)

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
