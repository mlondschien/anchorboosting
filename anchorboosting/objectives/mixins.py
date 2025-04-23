import numpy as np

from anchorboosting.utils import proj


class ProjMixin:
    def __init__(self, precompute_proj=False, n_categories=None):
        self.n_categories = n_categories
        self.precompute_proj = precompute_proj
        self.pinvZ = None

    def proj(self, Z, *args, copy=False):
        if self.n_categories is not None:
            return proj(Z, *args, n_categories=self.n_categories, copy=copy)
        elif self.precompute_proj:
            if self.pinvZ is None:
                self.pinvZ = np.linalg.pinv(Z)
            if len(args) == 1:
                return np.dot(Z, self.pinvZ @ args[0])
            else:
                return (*(np.dot(Z, self.pinvZ @ f) for f in args),)
        else:
            return proj(Z, *args, n_categories=None, copy=copy)


class LGBMMixin:
    higher_is_better = False

    def objective(self, f, data):
        """Objective function for LGBM."""
        return self.grad(f, data), self.hess(f, data)

    def score(self, f, data):
        """Score function for LGBM."""
        return (
            f"{self.name} ({self.gamma})",
            self.loss(f, data).mean(),
            self.higher_is_better,
        )


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


class ClassificationMixin:
    def init_score(self, y):
        """Initial score for LGBM.

        Parameters
        ----------
        y: np.ndarray of dimension (n,)
            Vector with true labels in (0, 1).

        Returns
        -------
        np.ndarray of length n
            Initial scores for LGBM.
        """
        p = np.sum(y) / len(y)
        return np.ones(len(y)) * np.log(p / (1 - p))

    def loss(self, f, data):
        """Two-class negative log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, 1).

        Returns
        -------
        np.ndarray of dimension (n,)
            Loss.
        """
        return np.log(1 + np.exp((-2 * data.get_label() + 1) * f))

    def predictions(self, f):
        """Compute probability predictions from scores via softmax.

        Parameters
        ----------
        f: np.ndarray of dimension (n,)
            Vector with scores.

        Returns
        -------
        np.ndarray of dimension (n,)
            Vector with probabilities.
        """
        return 1 / (1 + np.exp(-f))

    def grad(self, f, data):
        """
        Gradient of the two-class log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n,)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, 1).
        """
        return self.predictions(f) - data.get_label()

    def hess(self, f, data):
        """
        Diagonal of the Hessian of the multi-class log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n,)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, 1).
        """
        predictions = self.predictions(f)
        return predictions * (1.0 - predictions)


class MultiClassificationMixin:
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.factor = (n_classes - 1) / n_classes

    def init_score(self, y):
        """Initial score for LGBM.

        Parameters
        ----------
        y: np.ndarray of dimension (n,)
            Vector with true labels in (0, ..., n_classes - 1).

        Returns
        -------
        np.ndarray of dimension (n * n_classes,)
            Initial scores for LGBM. Note that this is flattened.
        """
        unique_values, unique_counts = np.unique(y, return_counts=True)
        assert len(unique_values) == self.n_classes
        assert (sorted(unique_values) == unique_values).all()

        odds = np.array(unique_counts) / np.sum(unique_counts)
        return np.log(np.tile(odds, (len(y), 1)).flatten("F"))

    def loss(self, f, data):
        """Multi-class negative log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n * n_classes,)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, ..., n_classes - 1).

        Returns
        -------
        np.ndarray of dimension (n,).
            Loss.
        """
        y = data.get_label()
        f = f.reshape((-1, self.n_classes), order="F")  # (n, n_classes)
        f = f - np.max(f, axis=1)[:, np.newaxis]  # normalize f to avoid overflow
        log_divisor = np.log(np.sum(np.exp(f), axis=1))
        return -f[self._indices(y)] + log_divisor

    def _indices(self, y):
        return (np.arange(len(y)), y.astype(int))

    def predictions(self, f):
        """Compute probability predictions from scores via softmax.

        Parameters
        ----------
        f: np.ndarray of dimension (n * n_classes,).]
            Vector with scores.

        Returns
        -------
        np.ndarray of dimension (n, n_classes)
            Vector with probabilities.
        """
        f = f.reshape((-1, self.n_classes), order="F")  # (n, n_classes)
        f = f - np.max(f, axis=1)[:, np.newaxis]  # normalize f to avoid overflow
        predictions = np.exp(f)
        predictions /= np.sum(predictions, axis=1)[:, np.newaxis]
        return predictions

    def grad(self, f, data):
        """
        Gradient of the multi-class log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n * n_classes,)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, ..., n_classes - 1).
        """
        y = data.get_label()
        predictions = self.predictions(f)
        predictions[self._indices(y)] -= 1

        return predictions.flatten("F")

    def hess(self, f, data):
        """
        Diagonal of the Hessian of the multi-class log-likelihood loss.

        Parameters
        ----------
        f: np.ndarray of dimension (n * n_classes,)
            Vector with scores.
        data: lgbm.Dataset
            LGBM dataset with labels of dimension (n,) in (0, ..., n_classes - 1).
        """
        predictions = self.predictions(f).flatten("F")
        return 1 / self.factor * predictions * (1.0 - predictions)
