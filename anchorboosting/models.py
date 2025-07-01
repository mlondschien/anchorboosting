import os
import sys
from contextlib import contextmanager
from logging import getLogger

import lightgbm as lgb
import numpy as np
import scipy

try:
    import polars as pl

    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False

logger = getLogger(__name__)


class AnchorBooster:
    """
    Boost the anchor loss.

    For regression, the anchor loss :cite:p:`rothenhausler2021anchor` with causal
    regularization parameter :math:`\\gamma` is

    .. math:: \\ell(f, y) = \\frac{1}{2} \\| y - f \\|_2^2 + \\frac{1}{2} (\\gamma - 1) \\|P_A (y - f) \\|_2^2,

    where :math:`P_A = A (A^T A)^{-1} A^T` is the linear projection onto the anchor
    :math:`A`'s column space .

    Let :math:`\\Phi` and :math:`\\varphi` be cumulative distribution function and
    probability density function of the Gaussian distribution.
    For binary classification with :math:`y \\in \\{-1, 1\\}` and a probit link
    function, the anchor loss :cite:p:`kook2022distributional` is

    .. math:: \\ell(f, y) = - \\sum_{i=1}^n \\log( \\Phi(y_i f_i) ) + \\frac{1}{2} (\\gamma - 1) \\|P_A r \\|_2^2,

    where :math:`r = - y \\varphi(f) / \\Phi(y f)` is the gradient of the probit loss
    :math:`- \\sum_{i=1}^n \\log( \\Phi(y_i f_i) )` with respect to the scores
    :math:`f`. We use a probit link instead of logistic as the resulting anchor loss is
    convex.

    We boost the anchor loss with LightGBM.
    Let :math:`\\hat f^j` be the boosted learner after :math:`j` steps of boosting, with
    :math:`\\hat f^0 = \\frac{1}{n} \\sum_{i=1}^n y_i` (regression) or
    :math:`\\hat f^0 = \\Phi^{-1}(\\frac{1}{n} \\sum_{i=1}^n y_i)` (binary classification).
    We fit a decision tree
    :math:`\\hat t^{j+1} := - \\left. \\frac{\\mathrm{d}}{\\mathrm{d} f} \\ell(f, y) \\right|_{f = \\hat f^j(X)} \\sim X`
    to the anchor loss' negative gradient.
    Let :math:`M \\in \\mathbb{R}^{n \\times \\mathrm{num. \\ leafs}}` be the one-hot encoding
    of :math:`\\hat t^{j+1}(X)`'s leaf node indices.
    Then
    :math:`M^T \\left. \\frac{\\mathrm{d}}{\\mathrm{d} f} \\ell(f, y) \\right|_{f = \\hat f^j(X)}`
    and
    :math:`M^T \\left.\\frac{\\mathrm{d}^2}{\\mathrm{d} f^2}\\ell(f, y)\\right|_{f = \\hat f^j(X)} M`
    are the gradient and Hessian of the loss function
    :math:`\\ell(\\hat f^j(X) + \\hat t^{j+1}(X), y) = \\ell(\\hat f^j(X) + M \\hat\\beta^{j+1}, y)`
    with respect to :math:`\\hat t^{j+1}`'s leaf node values
    :math:`\\hat\\beta^{j+1} \\in \\mathbb{R}^{\\mathrm{num. \\ leafs}}`.
    We set them using a second order optimization step

    .. math:: \\hat \\beta^{j+1} = - \\mathrm{lr} \\, \\cdot \\, \\left( M^T \\left.\\frac{\\mathrm{d}^2}{\\mathrm{d} f^2}\\ell(f, y)\\right|_{f = \\hat f^j(X)} M \\right)^{-1} M^T \\left.\\frac{\\mathrm{d}}{\\mathrm{d} f}\\ell(f, y)\\right|_{f = \\hat f^j(X)},

    where :math:`\\mathrm{lr}` is the learning rate, 0.1 by default.
    Finally, we set :math:`\\hat f^{j+1} = \\hat f^j + \\hat t^{j+1}`.

    For optimal speed, set the environment variable ``OMP_NUM_THREADS`` to the number of
    CPU cores available (not threads) before training. For performance, we recommend
    reducing the tree's variance by restricting their maximum depth or number of leaves,
    e.g., by setting ``max_depth=3``. Also, consider setting ``min_gain_to_split=0.1``
    (or some other small, non-zero value) to keep LightGBM from splitting leaves with
    zero variance.

    Parameters
    ----------
    gamma: float
        The :math:`\\gamma` parameter for the anchor objective function. Must be
        non-negative. If 1, the objective is equivalent to a standard regression or
        probit classification objective.
        Larger values correspond to more causal regularization.
    dataset_params: dict or None
        The parameters for the LightGBM dataset. See LightGBM documentation for details.
        If None, LightGBM defaults are used.
    num_boost_round: int
        The number of boosting iterations. Default is 100.
    objective: str, optional, default="regression"
        The objective function to use. Can be ``"regression"`` for regression or
        ``"binary"`` for classification with a probit link function. If ``"binary"``,
        the outcome values must be 0 or 1.
    learning_rate: float, optional, default=0.1
        The learning rate for the boosting. This is the :math:`\\mathrm{lr}` in the
        second order optimization step. It controls the step size of the updates.
    **kwargs: dict
        Additional parameters for the LightGBM model. See LightGBM documentation for
        details. We suggest reducing the tree's complexity by reducing ``max_depth`` or
        ``num_leaves`` and setting ``min_gain_to_split`` to a non-zero value.

    Attributes
    ----------
    booster_: lightgbm.Booster
        The LightGBM booster containing the trained model.
    init_score_: float
        The initial score used for the boosting. For regression, this is the mean of
        the outcome values. For binary classification, this is the inverse probit link
        applied to the prevalence.

    References
    ----------
    .. bibliography::
       :filter: False

       rothenhausler2021anchor
       kook2022distributional
    """  # noqa: E501

    def __init__(
        self,
        gamma,
        dataset_params=None,
        num_boost_round=100,
        objective="regression",
        learning_rate=0.1,
        **kwargs,
    ):
        self.gamma = gamma
        kwargs["learning_rate"] = learning_rate
        kwargs["objective"] = "none"

        self.params = kwargs
        self.dataset_params = dataset_params
        self.num_boost_round = num_boost_round
        self.objective = objective

    def fit(
        self,
        X,
        y,
        Z=None,
        categorical_feature=None,
    ):
        """
        Fit the ``AnchorBooster``.

        Parameters
        ----------
        X : ``pl.DataFrame`` or ``np.ndarray`` or ``pyarrow.Table`` or ``pd.DataFrame``
            The input data.
        y : np.ndarray
            The outcome.
        Z : np.ndarray
            Anchors. One-hot encode categorical anchors.
        categorical_feature : list of str or int or None, optional
            List of categorical feature names or indices. If ``None``, all features are
            assumed to be numerical.

        Returns
        -------
        self : AnchorBooster
        """
        if self.objective != "regression" and not np.isin(y, [0, 1]).all():
            raise ValueError("For binary classification, y values must be in {0, 1}.")

        y = y.flatten()

        if self.objective == "regression":
            self.init_score_ = np.mean(y)
        elif self.objective == "binary":
            y_mean = np.clip(np.mean(y), 1 / len(y), 1 - 1 / len(y))
            self.init_score_ = scipy.stats.norm.ppf(y_mean)
        else:
            raise ValueError(
                f"Objective must be 'regression' or 'binary'. Got {self.objective}."
            )

        if hasattr(X, "columns"):
            feature_name = X.columns

        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            X = X.to_arrow()

        dataset_params = {
            "data": X,
            "label": y,
            "categorical_feature": categorical_feature,
            "feature_name": feature_name,
            "init_score": np.ones(len(y), dtype=np.float64) * self.init_score_,
            **(self.dataset_params or {}),
        }

        data = lgb.Dataset(**dataset_params)

        self.booster_ = lgb.Booster(params=self.params, train_set=data)

        f = np.ones(len(y), dtype=np.float64) * self.init_score_
        booster_preds = f.copy()

        proj = Proj(Z)

        if self.objective == "binary":
            y_tilde = np.where(y == 1, 1, -1).astype(np.float64)

        max_num_leaves = self.params.get("num_leaves", 31)
        max_depth = self.params.get("max_depth", None)
        if max_depth is not None and max_depth > 0:
            max_num_leaves = min(2**max_depth, max_num_leaves)

        for idx in range(self.num_boost_round):
            # For regression, the loss (without anchor) is
            # loss(f, y) = 0.5 * || y - f ||^2
            if self.objective == "regression":
                r = f - y  # d/df loss(f, y)
                dr = np.ones(len(y), dtype=np.float64)  # d^2/df^2 loss(f, y)
                ddr = np.zeros(len(y), dtype=np.float64)  # d^3/df^3 loss(f, y)
            # For probit regression, the loss (without anchor) is
            # loss(f, y) = - sum_i (y_i log(p_i) + (1 - y_i) log(1 - p_i))
            # where p_i = scipy.stats.cdf(f_i)
            else:
                # We wish to compute the following:
                # p = scipy.stats.norm.cdf(f)
                # dp = scipy.stats.norm.pdf(f)  # d/df p(f)
                # r = np.where(y == 1, -dp / p, dp / (1 - p))  # d/df loss(f, y)
                # The equation for r is numerically unstable. Instead, we use
                # scipy.special.log_ndtr, with log_ndtr(f) = log(norm.cdf(f)).
                log_phi = -0.5 * f**2 - 0.5 * np.log(2 * np.pi)  # log(norm.pdf(f))
                log_ndtr = scipy.special.log_ndtr(y_tilde * f)
                r = -y_tilde * np.exp(log_phi - log_ndtr)
                dr = -f * r + r**2  # d^2/df^2 loss(f, y)
                ddr = (f**2 - 1) * r - 3 * f * r**2 + 2 * r**3  # d^3/df^3 loss

            r_proj = proj(r)
            grad = r + (self.gamma - 1) * r_proj * dr

            # We wish to fit one additional tree. Intuitively, one would use
            # is_finished = self.booster_.update(fobj=self.objective.objective)
            # for this. This makes a call to self.__inner_predict(0) to get the current
            # predictions for all existing trees. See:
            # https://github.com/microsoft/LightGBM/blob/18c11f861118aa889b9d4579c2888d\
            # 5c908fd250/python-package/lightgbm/basic.py#L4165
            # To avoid passing data through all trees each time, this uses a cache.
            # However, this cache is based on the "original" tree values, not the one
            # we set below. We thus use "our own" predictions and skip __inner_predict.
            # No idea what the set_objective_to_none does, but lgbm raises if we don't.
            self.booster_._Booster__inner_predict_buffer = None
            if not self.booster_._Booster__set_objective_to_none:
                self.booster_.reset_parameter(
                    {"objective": "none"}
                )._Booster__set_objective_to_none = True

            # is_finished is True if there we no splits satisfying the splitting
            # criteria. c.f. https://github.com/microsoft/LightGBM/pull/6890
            # The hessian is used only for the `min_hessian_in_leaf` parameter to
            # avoid numerical instabilities.
            is_finished = self.booster_._Booster__boost(grad, dr)

            if is_finished:
                logger.info(f"Finished training after {idx} iterations.")
                break

            # We recover the leaf indices of the current tree, avoiding (slow) call to
            # self.booster_.predict(X, pred_leaf=True). It's a bit of dark magic, but
            # the speedup is worth it.
            # leaves = self.booster_.predict(
            #     X, start_iteration=idx, num_iteration=1, pred_leaf=True
            # ).flatten()
            leaf_values = []
            num_leaves = max_num_leaves

            # There exists no "booster.get_num_leafs(idx)" in LGBM. One could use
            # num_leaves = self.booster_.dump_model()["tree_info"][idx]["num_leaves"],
            # but this is slow. The try-except loop below is faster.
            # Even though we catch the error, LightGBM prints to stderr somewhere in the
            # C-code. We catch and suppress this.
            for ldx in range(max_num_leaves):
                try:
                    with _suppress_stderr():
                        val = self.booster_.get_leaf_output(idx, ldx)
                    leaf_values.append(val)
                except lgb.basic.LightGBMError:
                    num_leaves = ldx
                    break

            leaf_values = np.array(leaf_values, dtype=np.float64)

            # The _Booster__inner_predict(0) checks if __inner_predict_buffer[0] is None
            self.booster_._Booster__inner_predict_buffer = [None]
            booster_preds_new = self.booster_._Booster__inner_predict(0)

            booster_preds_this_iter = booster_preds_new - booster_preds
            booster_preds = booster_preds_new

            leaf_values_argsort = np.argsort(leaf_values)
            leaf_values_sorted = leaf_values[leaf_values_argsort]

            tol = np.min(np.abs(np.diff(leaf_values_sorted))) * 0.1
            # If min_gain_to_split=0, LightGBM will split a leaf with zero variance
            # (e.g., all same class in binary classification during first boosting
            # iteration with f=const) into two leaves. Then, there are 2 leafs with
            # exactly the same value and our approach above cannot differentiate them.
            if tol < 1e-12:
                logger.info(
                    "LightGBM split a leaf with (almost) zero variance at iteration "
                    f"{idx}. Consider increasing `min_gain_to_split`."
                )
                leaves = self.booster_.predict(
                    X, start_iteration=idx, num_iteration=1, pred_leaf=True
                ).flatten()
            else:
                # searchsorted finds indices i such that
                # leaf_values_sorted[i-1] + tol < preds <=leaf_values_sorted[i] + tol.
                leaves = np.searchsorted(
                    leaf_values_sorted + tol,
                    booster_preds_this_iter,
                    side="left",
                )
                leaves = leaf_values_argsort[leaves]

            # We wish to do 2nd order updates in the leaves. Since the anchor regression
            # objective is quadratic, for regression a 2nd order update is equal to the
            # global minimizer.
            # Let M be the one-hot encoding of the tree's leaf assignments. That is,
            # M[i, j] = 1 if leaves[i] == j else 0.
            # We have
            # r = d/df loss(f, y)
            # dr = d^2/df^2 loss(f, y)
            # ddr = d^3/df^3 loss(f, y)
            # The anchor loss is
            # L = loss(f, y) + (gamma - 1) / 2 * || P_Z r ||^2
            # d/df L = d/df loss(f, y) + (gamma - 1) P_Z r * dr
            # d^2/df^2 L = diag(d^2/df^2 loss(f, y)) + (gamma - 1) diag(P_Z r * ddr)
            #            + (gamma - 1) diag(dr) P_Z diag(dr)
            #
            # We do the 2nd order update
            # beta = - (M^T [d^2/df^2 L] M)^{-1} M^T [d/df L]

            # M^T x = bincount(leaves, weights=x)
            g = np.bincount(leaves, weights=grad, minlength=num_leaves)

            # M^T diag(x) M = diag(np.bincount(leaves, weights=x))
            counts = np.bincount(
                leaves,
                weights=dr + (self.gamma - 1) * r_proj * ddr,
                minlength=num_leaves,
            )
            counts += self.params.get("lambda_l2", 0)
            counts[counts == 0] = 1  # counts==0 should never happen.

            # (dr * M)^T P_Z (dr * M)
            H = (self.gamma - 1) * proj.sandwich(leaves, num_leaves, weights=dr)
            H += np.diag(counts)

            # Compute the 2nd order update
            leaf_values = -np.linalg.solve(H, g) * self.params.get("learning_rate", 0.1)

            for ldx, val in enumerate(leaf_values):
                self.booster_.set_leaf_output(idx, ldx, val)

            # Ensure f == self.init_score_ + self.booster_.predict(X)
            f += leaf_values[leaves]

        return self

    def predict(self, X, raw_score=False, **kwargs):
        """
        Predict the outcome.

        Parameters
        ----------
        X : numpy.ndarray, polars.DataFrame, or pyarrow.Table
            The input data.
        raw_score : bool
            If ``True``, returns scores. Returns predicted probabilities if
            ``objective`` is ``"binary"`` and ``raw_score`` is ``False``.
        kwargs : dict
            Passed to ``lgb.Booster.predict``.
        """
        if not hasattr(self, "booster_"):
            raise ValueError("AnchorBooster has not yet been fitted.")

        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            X = X.to_arrow()

        scores = self.booster_.predict(X, raw_score=True, **kwargs)

        if self.objective == "binary" and not raw_score:
            return scipy.stats.norm.cdf(scores + self.init_score_)
        else:
            return scores + self.init_score_

    def refit(self, X, y, decay_rate=0):
        """
        Refit the model using new data.

        Set :math:`\\hat f^0_\\mathrm{refit} =` ``self.init_score_``.
        Starting from :math:`\\hat f^j_\\mathrm{refit}`, we drop the new data
        :math:`(X, y)` down the :math:`j + 1`'th tree :math:`\\hat t^{j+1}`.
        Let :math:`\\hat \\beta_\\mathrm{new}^{j+1}` be the second order optimization
        of the loss :math:`\\ell(\\hat f^j_\\mathrm{refit} + \\hat t^{j+1}(X), y)`
        with respect to the leaf node values :math:`\\beta^{j+1}` of
        :math:`\\hat t^{j+1}(X)`.
        We set
        :math:`\\hat \\beta^{j+1}_\\mathrm{refit} = \\mathrm{decay \\ rate} \\cdot \\hat \\beta^{j+1}_\\mathrm{old} + (1 - \\mathrm{decay \\ rate}) \\cdot \\hat \\beta^{j+1}_\\mathrm{new}`.
        Refitting updates the tree's leaf values, but not their structure.
        ``AnchorBooster.refit`` differs from ``lgbm.Booster.refit`` by not reestimating
        :math:`\\hat f^0_\\mathrm{refit}` from the new :math:`y`, supporting
        probit regression, and by not updating leaf node values with no samples from the
        new data, instead of shrinking them towards zero.

        Parameters
        ----------
        X : numpy.ndarray, polars.DataFrame, or pyarrow.Table
            The new data.
        y : np.ndarray
            The new outcomes.
        decay_rate : float
            The decay rate for the leaf values. Must be in [0, 1]. Default is 0. If 0,
            the leaf values are set to the new values. If 1, the leaf values are not
            updated. This matches the behavior of LightGBM's ``refit`` method.

        Returns
        -------
        self: AnchorBooster
        """  # noqa: E501
        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            X = X.to_arrow()

        if self.objective == "binary" and not np.isin(y, [0, 1]).all():
            raise ValueError("For binary classification, y values must be in {0, 1}.")

        if self.objective == "binary":
            y_tilde = np.where(y == 1, 1, -1)

        leaves = self.booster_.predict(X, pred_leaf=True)
        num_leaves = np.max(leaves, axis=0) + 1

        f = np.full(len(y), self.init_score_, dtype="float64")

        # The model might have stopped boosting prematurely.
        for idx in range(leaves.shape[1]):
            n_obs = np.bincount(leaves[:, idx], minlength=num_leaves[idx])

            if self.objective == "regression":
                sum_grad = np.bincount(
                    leaves[:, idx],
                    weights=f - y,  # grad
                    minlength=num_leaves[idx],
                )
                sum_hess = np.clip(n_obs, 1.0, None)

            elif self.objective == "binary":
                log_phi = -0.5 * f**2 - 0.5 * np.log(2 * np.pi)  # log(norm.pdf(f))
                grad = -y_tilde * np.exp(log_phi - scipy.special.log_ndtr(y_tilde * f))
                sum_grad = np.bincount(
                    leaves[:, idx], weights=grad, minlength=num_leaves[idx]
                )
                sum_hess = np.bincount(
                    leaves[:, idx],
                    weights=-f * grad + grad**2,  # hess
                    minlength=num_leaves[idx],
                )
                sum_hess = np.clip(sum_hess, 1.0, None)
            else:
                raise ValueError("Objective must be 'regression' or 'binary'.")

            values = -sum_grad / sum_hess * self.params.get("learning_rate", 0.1)

            new_values = np.zeros(num_leaves[idx], dtype="float64")

            for ldx in np.where(n_obs > 0)[0]:
                old_value = self.booster_.get_leaf_output(idx, ldx)
                new_values[ldx] = (
                    decay_rate * old_value + (1 - decay_rate) * values[ldx]
                )
                self.booster_.set_leaf_output(idx, ldx, new_values[ldx])

            f += new_values[leaves[:, idx]]

        return self


@contextmanager
def _suppress_stderr():
    stderr_fd = sys.stderr.fileno()  # 2
    old_stderr_fd = os.dup(stderr_fd)  # keep to restore later

    devnull_fd = os.open(os.devnull, os.O_RDWR)  # open /dev/null for writing

    try:
        os.dup2(devnull_fd, stderr_fd)  # redirect 2 → /dev/null
        yield  # run my code
    finally:
        os.dup2(old_stderr_fd, stderr_fd)  # put the old stderr back
        os.close(old_stderr_fd)
        os.close(devnull_fd)


class Proj:
    """
    Cache the projection onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z) or (n,), optional, default=None
        The `Z` matrix or 1d array of integers.
    """

    def __init__(self, Z):
        if Z is None:
            self._type = "none"

        elif len(Z.shape) == 1 and np.issubdtype(Z.dtype, np.integer):
            self._type = "categorical"
            self._Z = Z
            self._n_categories = np.max(Z) + 1
            counts = np.bincount(Z, minlength=self._n_categories)
            counts[counts == 0] = 1.0  # Avoid division by zero
            self._one_over_counts = 1 / counts

        elif len(Z.shape) == 2 and np.issubdtype(Z.dtype, np.floating):
            self._type = "linear"
            self._Q, _ = np.linalg.qr(Z, mode="reduced")
        else:
            raise ValueError(
                "Z should be either a 1d array of integers or a 2d array of floats. "
                f"Got shape {Z.shape} and dtype {Z.dtype}."
            )

    def __call__(self, f):
        """
        Project the input array `f` onto the subspace spanned by `Z`.

        Parameters
        ----------
        f: np.ndarray of shape (n,)
            The input array to project.

        Returns
        -------
        np.ndarray of shape (n,)
            The projected array.
        """
        if self._type == "none":
            return np.zeros_like(f)

        elif self._type == "categorical":
            sums = np.bincount(self._Z, weights=f, minlength=self._n_categories)
            means = sums * self._one_over_counts
            return means[self._Z]

        elif self._type == "linear":
            return self._Q @ (self._Q.T @ f)

    def sandwich(self, leaves, num_leaves, weights):
        """
        For M = weights * one_hot(leaves), return proj(Z, M).T @ proj(Z, M).

        Parameters
        ----------
        leaves: np.ndarray of shape (n,)
            The leaf indices for each sample in `f`. Integers in [0, num_leaves).
        num_leaves: int
            The number of leaves in the decision tree.
        weights: np.ndarray of shape (n,)
            The input array to project.

        Returns
        -------
        np.ndarray of shape (d, d)
            The sandwich product.
        """
        if self._type == "none":
            return np.zeros((num_leaves, num_leaves), dtype=weights.dtype)

        elif self._type == "categorical":
            S = np.zeros((self._n_categories, num_leaves), dtype=weights.dtype)
            np.add.at(S, (self._Z, leaves), weights)
            return (S * self._one_over_counts[:, np.newaxis]).T @ S

        elif self._type == "linear":
            # M^T P_Z M = (M^T Q) @ (M^T Q)^T
            # One could also compute this using bincount, but it appears this
            # version using a sparse matrix is faster.
            M = scipy.sparse.csr_matrix(
                (
                    weights,
                    (np.arange(len(leaves)), leaves),
                ),
                shape=(len(leaves), num_leaves),
                dtype=weights.dtype,
            )
            B = M.T @ self._Q
            return B @ B.T
