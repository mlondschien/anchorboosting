import copy

import lightgbm as lgb
import numpy as np
import scipy

try:
    import polars as pl

    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False


class AnchorBooster:
    """
    Boost the anchor loss.

    Parameters
    ----------
    gamma: float
        The gamma parameter for the anchor regression objective function. Must be non-
        negative. If 1, the objective is equivalent to a standard regression objective.
    dataset_params: dict or None
        The parameters for the LightGBM dataset. See LightGBM documentation for details.
    num_boost_round: int
        The number of boosting iterations. Default is 100.
    objective: str, optional, default="regression"
        The objective function to use. Can be "regression" or "binary" for probit
        regression. If "binary", the outcome values must be 0 or 1.
    **kwargs: dict
        Additional parameters for the LightGBM model. See LightGBM documentation for
        details.
    """

    def __init__(
        self,
        gamma,
        dataset_params=None,
        num_boost_round=100,
        objective="regression",
        **kwargs,
    ):
        self.gamma = gamma
        self.params = kwargs
        self.dataset_params = dataset_params or {}
        self.num_boost_round = num_boost_round
        self.booster = None
        self.init_score_ = None
        self.objective = objective

    def fit(
        self,
        X,
        y,
        Z,
        categorical_feature=None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X : polars.DataFrame
            The input data.
        y : np.ndarray
            The outcome.
        Z : np.ndarray
            Anchors.
        categorical_feature : list of str or int
            List of categorical feature names or indices. If None, all features are
            assumed to be numerical.
        """
        if self.objective != "regression" and not np.isin(y, [0, 1]).all():
            raise ValueError("For binary classification, y values must be in {0, 1}.")

        if self.objective == "regression":
            self.init_score_ = np.mean(y)
        elif self.objective == "binary":
            self.init_score_ = scipy.stats.norm.ppf(np.mean(y))
        else:
            raise ValueError(
                f"Objective must be 'regression' or 'binary'. Got {self.objective}."
            )

        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            feature_name = X.columns
            X = X.to_arrow()
        else:
            feature_name = None

        self._dtype = np.result_type(Z, y)

        dataset_params = {
            "data": X,
            "label": y,
            "categorical_feature": categorical_feature,
            "feature_name": feature_name,
            "init_score": np.ones(len(y), dtype=self._dtype) * self.init_score_,
            **self.dataset_params,
        }

        data = lgb.Dataset(**dataset_params)

        self.booster = lgb.Booster(params=self.params, train_set=data)

        return self.update(
            X,
            y,
            Z=Z,
            num_iteration=self.num_boost_round,
        )

    def update(self, X, y, Z, num_iteration=1):
        if self.booster is None or self.init_score_ is None:
            raise ValueError("AnchorBoost has not yet been fitted.")

        y = y.flatten()

        current_iteration = self.booster.current_iteration()
        if current_iteration == 0:
            f = np.ones(len(y), dtype=self._dtype) * self.init_score_
        else:
            f = self.booster.predict(X, raw_score=True) + self.init_score_

        Q, _ = np.linalg.qr(Z, mode="reduced")  # P_Z f = Q @ (Q^T @ f)

        for idx in range(current_iteration, current_iteration + num_iteration):
            # For regression, the loss (without anchor) is
            # loss(f, y) = 0.5 * || y - f ||^2
            if self.objective == "regression":
                r = f - y  # d/df loss(f, y)
                dr = np.ones(len(y), dtype=self._dtype)  # d^2/df^2 loss(f, y)
                ddr = np.zeros(len(y), dtype=self._dtype)  # d^3/df^3 loss(f, y)
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
                y_tilde = np.where(y == 1, 1, -1)
                log_phi = -0.5 * f**2 - 0.5 * np.log(2 * np.pi)  # log(norm.pdf(f))
                r = -y_tilde * np.exp(log_phi - scipy.special.log_ndtr(y_tilde * f))
                dr = -f * r + r**2  # d^2/df^2 loss(f, y)
                ddr = (f**2 - 1) * r - 3 * f * r**2 + 2 * r**3  # d^3/df^3 loss

            r_proj = Q @ (Q.T @ r)
            grad = r + (self.gamma - 1) * r_proj * dr

            # We wish to fit one additional tree. Intuitively, one would use
            # is_finished = self.booster.update(fobj=self.objective.objective)
            # for this. This makes a call to self.__inner_predict(0) to get the current
            # predictions for all existing trees. See:
            # https://github.com/microsoft/LightGBM/blob/18c11f861118aa889b9d4579c2888d\
            # 5c908fd250/python-package/lightgbm/basic.py#L4165
            # To avoid passing data through all trees each time, this uses a cache.
            # However, this cache is based on the "original" tree values, not the one
            # we set below. We thus use "our own" predictions and skip __inner_predict.
            # No idea what the set_objective_to_none does, but lgbm raises if we don't.
            self.booster._Booster__inner_predict_buffer = None
            if not self.booster._Booster__set_objective_to_none:
                self.booster.reset_parameter(
                    {"objective": "none"}
                )._Booster__set_objective_to_none = True

            # is_finished is True if there we no splits satisfying the splitting
            # criteria. c.f. https://github.com/microsoft/LightGBM/pull/6890
            # The hessian is used only for the `min_hessian_in_leaf` parameter to
            # avoid numerical instabilities.
            is_finished = self.booster._Booster__boost(grad, dr)

            if is_finished:
                print(f"Finished training after {idx} iterations.")
                break

            leaves = self.booster.predict(
                X, start_iteration=idx, num_iteration=1, pred_leaf=True
            ).flatten()
            num_leaves = np.max(leaves) + 1

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

            # M^T x = bincount(leaves_masked, weights=x)
            g = np.bincount(leaves, weights=grad, minlength=num_leaves)

            # M^T diag(x) M = diag(np.bincount(leaves, weights=x))
            counts = np.bincount(
                leaves,
                weights=dr + (self.gamma - 1) * r_proj * ddr,
                minlength=num_leaves,
            )
            counts += self.params.get("lambda_l2", 0)

            # Mdr^T P_Z Mdr = (Mdr^T Q) @ (Mdr^T Q)^T
            # One could also compute this using bincount, but it appears this
            # version using a sparse matrix is faster.
            Mdr = scipy.sparse.csr_matrix(
                (
                    dr,
                    (np.arange(len(leaves)), leaves),
                ),
                shape=(len(leaves), num_leaves),
                dtype=self._dtype,
            )
            B = Mdr.T.dot(Q)
            H = (self.gamma - 1) * B @ B.T
            H += np.diag(counts)

            # Compute the 2nd order update
            leaf_values = -np.linalg.solve(H, g) * self.params.get("learning_rate", 0.1)

            for ldx, val in enumerate(leaf_values):
                self.booster.set_leaf_output(idx, ldx, val)

            # Ensure f == self.init_score_ + self.booster.predict(X)
            f += leaf_values[leaves]

        return self

    def predict(self, X, raw_score=False, **kwargs):
        """
        Predict the outcome.

        Parameters
        ----------
        X : numpy.ndarray, polars.DataFrame, or pyarrow.Table
            The input data.
        num_iteration : int
            Number of boosting iterations to use. If -1, all are used. Else, needs to be
            in [0, num_boost_round].
        raw_score : bool
            If True, returns scores. If False, returns predicted probabilities.
        """
        if self.booster is None:
            raise ValueError("AnchorBoost has not yet been fitted.")

        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            X = X.to_arrow()

        scores = self.booster.predict(X, raw_score=True, **kwargs)

        if self.objective == "binary" and not raw_score:
            return scipy.stats.norm.cdf(scores + self.init_score_)
        else:
            return scores + self.init_score_

    def refit(self, X, y, decay_rate=0):
        """
        Refit the model using new data.

        Set :math:`f^0_\\mathrm{refit} =` ``init_score_``.
        Starting from :math:`f^j_\\mathrm{refit}`, we drop the new data :math:`(X, y)`
        down the tree :math:`\\hat t^{j+1}`.
        Let :math:`\\hat \\beta_\\mathrm{new}^{j+1}` be the second order optimization
        of the loss :math:`\\ell(\\hat f^j_\\mathrm{refit} + \\hat t^{j+1}(X), y)`
        with respect to the leaf node values :math:`\\beta^{j+1}``of
        :math:`\\hat t^{j+1}(X)`.
        We set
        :math:`\\hat \\beta^{j+1}_\\mathrm{refit} = \\mathrm{decay rate} \\hat \\beta^{j+1}_\\mathrm{old} + (1 - \\mathrm{decay rate}) \\hat \\beta^{j+1}_\\mathrm{new}`.
        Refitting updates the tree's leaf values, but not their structure.
        ``AnchorBooster.refit`` differs from ``lgbm.Booster.refit`` in that it supports
        probit regression and leaf nodes with no samples from the new data are not,
        updated, instead of being shrunk towards zero (as in LightGBM).

        Refit is not in-place, but returns a new instance of ``AnchorBooster``.

        Parameters
        ----------
        X : numpy.ndarray, polars.DataFrame, or pyarrow.Table
            The new data.
        y : np.ndarray
            The new outcomes.
        decay_rate : float
            The decay rate for the leaf values. Must be in [0, 1]. Default is 0. If 0,
            the leaf values are set to the new values. If 1, the leaf values are not
            updated.

        Returns
        -------
        AnchorBooster
            A new instance of AnchorBooster with the updated leaf values.
        """  # noqa: E501
        self_copied = copy.deepcopy(self)

        # For some reason, the model params are not copied over.
        # https://github.com/microsoft/LightGBM/issues/6821
        self_copied.booster.params = self.booster.params

        if self.objective == "binary" and not np.isin(y, [0, 1]).all():
            raise ValueError("For binary classification, y values must be in {0, 1}.")

        if self.objective == "binary":
            y_tilde = np.where(y == 1, 1, -1)

        leaves = self_copied.booster.predict(X, pred_leaf=True)
        f = np.full(len(y), self.init_score_, dtype="float64")

        for idx in range(self.num_boost_round):
            if self.objective == "regression":
                grad_hess_ones = np.empty((len(y), 3), dtype="float64")
                grad_hess_ones[:, 0] = f - y
                grad_hess_ones[:, 1:] = 1.0
                # g = f - y
                # h = 1.0
            elif self.objective == "binary":
                log_phi = -0.5 * f**2 - 0.5 * np.log(2 * np.pi)  # log(norm.pdf(f))
                grad_hess_ones = np.empty((len(y), 3), dtype="float64")
                grad_hess_ones[:, 0] = -y_tilde * np.exp(
                    log_phi - scipy.special.log_ndtr(y_tilde * f)
                )
                grad_hess_ones[:, 1] = (
                    -f * grad_hess_ones[:, 0] + grad_hess_ones[:, 0] ** 2
                )
                grad_hess_ones[:, 2] = 1.0
                # g = -y_tilde * np.exp(log_phi - scipy.special.log_ndtr(y_tilde * f))
                # h = -f * g + g**2
            else:
                raise ValueError("Objective must be 'regression' or 'binary'.")

            num_leaves = np.max(leaves[:, idx]) + 1

            # We aggregate gradients, hessians, and counts over the leaves.
            # A bincount is faster than a for loop. Still this passes 3x through the
            # data. We use np.add.at to do this in a single pass and avoid a copy.
            # n_obs = np.bincount(leaves[:, idx], minlength=num_leaves)
            # sum_grad = np.bincount(leaves[:, idx], weights=g, minlength=num_leaves)
            # sum_hess = np.bincount(leaves[:, idx], weights=h, minlength=num_leaves)
            arr = np.zeros((num_leaves, 3), dtype=np.float64)
            np.add.at(arr, leaves[:, idx], grad_hess_ones)

            sum_grad = arr[:, 0]
            sum_hess = arr[:, 1]
            n_obs = arr[:, 2]

            values = -sum_grad / sum_hess * self_copied.params.get("learning_rate", 0.1)

            new_values = np.zeros(num_leaves, dtype="float64")

            for ldx in np.where(n_obs > 0)[0]:
                old_value = self_copied.booster.get_leaf_output(idx, ldx)
                new_values[ldx] = (
                    decay_rate * old_value + (1 - decay_rate) * values[ldx]
                )
                self_copied.booster.set_leaf_output(idx, ldx, new_values[ldx])

            f += new_values[leaves[:, idx]]

        return self_copied
