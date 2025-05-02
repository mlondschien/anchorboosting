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
    Boost the anchor regression loss.

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
        The objective function to use. Can be "regression", "logistic", or "probit".
    honest_splits: bool, optional, default=False
        If True, uses a different part of the training data to compute tree splits and
        leaf values.
    honest_splits_ratio: float, optional, default=0.5
        Proportion of the training data to use to compute tree splits when honest_splits
        is True. Must be in (0, 1).
    subsample: float, optional, default=1.0
        The fraction of samples to use for each boosting iteration. Must be in (0, 1].
        Acts multiplicatively with ``honest_splits_ratio``. Not supported when
        ``honest_splits`` is False.
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
        honest_splits=False,
        honest_splits_ratio=0.5,
        subsample=1.0,
        **kwargs,
    ):
        self.gamma = gamma
        self.params = kwargs
        self.dataset_params = dataset_params or {}
        self.num_boost_round = num_boost_round
        self.booster = None
        self.init_score_ = None
        self.objective = objective
        self.honest_splits = honest_splits
        self.honest_splits_ratio = honest_splits_ratio
        self.subsample = subsample

    def fit(
        self,
        X,
        y,
        Z=None,
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
        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            feature_name = X.columns
            X = X.to_arrow()
        else:
            feature_name = None

        dtype = np.result_type(Z, y)

        if self.objective == "regression":
            self.init_score_ = np.mean(y)
        elif self.objective == "probit":
            self.init_score_ = scipy.stats.norm.ppf(np.mean(y))
        else:
            self.init_score_ = np.log(np.mean(y) / (1 - np.mean(y)))

        if self.objective != "regression" and not np.isin(y, [0, 1]).all():
            raise ValueError("For binary classification, y values must be in {0, 1}.")

        y = y.flatten()

        dataset_params = {
            "data": X,
            "label": y,
            "categorical_feature": categorical_feature,
            "feature_name": feature_name,
            "init_score": np.ones(len(y), dtype=dtype) * self.init_score_,
            **self.dataset_params,
        }

        data = lgb.Dataset(**dataset_params)

        self.booster = lgb.Booster(params=self.params, train_set=data)
        f = dataset_params["init_score"]  # scores

        Q, _ = np.linalg.qr(Z, mode="reduced")  # P_Z f = Q @ (Q^T @ f)

        if self.honest_splits:
            n_split = int(len(y) * self.honest_splits_ratio * self.subsample)

            split_mask = np.zeros(len(y), dtype=bool)
            split_mask[:n_split] = True
            leaf_mask = np.zeros(len(y), dtype=bool)
            leaf_mask[n_split : int(len(y) * self.subsample)] = True
            rng = np.random.default_rng(0)

        for idx in range(self.num_boost_round):
            if self.honest_splits:
                perm = rng.permutation(len(y))
                split_mask = split_mask[perm]
                leaf_mask = leaf_mask[perm]

            # For regression, the loss (without anchor) is
            # loss(f, y) = 0.5 * || y - f ||^2
            if self.objective == "regression":
                r = f - y  # d/df loss(f, y)
                dr = np.ones(len(y), dtype=dtype)  # d^2/df^2 loss(f, y)
                ddr = np.zeros(len(y), dtype=dtype)  # d^3/df^3 loss(f, y)
            # For logistic regression, the loss (without anchor) is
            # loss(f, y) = - sum_i (y_i log(p_i) + (1 - y_i) log(1 - p_i))
            elif self.objective == "logistic":
                p = scipy.special.expit(f)
                r = p - y  # score residuals: r(f, y) = d/df loss(f, y)
                dr = p * (1 - p)  # d/df r(f, y) = d^2/df^2 loss(f, y)
                ddr = p * (1 - p) * (1 - 2 * p)  # d^3/df^3 loss(f, y)
            # For probit regression, the loss (without anchor) is
            # loss(f, y) = - sum_i (y_i log(p_i) + (1 - y_i) log(1 - p_i))
            # where p_i = P(f > 0) (Gaussian cdf)
            elif self.objective == "probit":
                p = scipy.stats.norm.cdf(f)
                dp = scipy.stats.norm.pdf(f)  # d/df p(f)
                A = np.where(y == 1, -1 / p, 1 / (1 - p))
                r = A * dp  # d/df loss(f, y)
                dr = -f * dp * A + dp**2 * A**2  # d^2/df^2 loss(f, y)
                ddr = (
                    (f**2 - 1) * dp * A
                    - 3 * f * dp**2 * A**2
                    + 2 * dp**3 * A**3
                )
            else:
                raise ValueError(
                    "Objective must be one of 'regression', 'logistic', or 'probit'. "
                    f" Got {self.objective}."
                )

            if self.honest_splits:
                Q_ = Q[split_mask, :]
                r_ = r[split_mask]
                dr_ = dr[split_mask]
                # Q_ @ (Q_.T @ r_) is Z[mask, :] @ (Z.T @ Z) @ (Z[mask, :]^T @ r_)
                # That is, we use the full Z matrix to compute its covariance. However,
                # different number of samples are used, so we need to rescale.
                r_proj = Q_ @ (Q_.T @ r_) / (self.honest_splits_ratio * self.subsample)

                grad = np.zeros(len(y), dtype=dtype)
                grad[split_mask] = r_ + (self.gamma - 1) * r_proj * dr_
                hess = np.zeros(len(y), dtype=dtype)
                hess[split_mask] = dr_

            else:
                r_proj = Q @ (Q.T @ r)
                grad = r + (self.gamma - 1) * r_proj * dr
                hess = dr

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
            is_finished = self.booster._Booster__boost(grad, hess)

            if is_finished:
                print(f"Finished training after {idx} iterations.")
                break

            leaves = self.booster.predict(
                X, start_iteration=idx, num_iteration=1, pred_leaf=True
            ).flatten()
            num_leaves = np.max(leaves) + 1

            if self.honest_splits:
                r_ = r[leaf_mask]
                dr_ = dr[leaf_mask]
                Q_ = Q[leaf_mask, :]
                # Q_ @ (Q_.T @ r_) is Z[mask, :] @ (Z.T @ Z) @ (Z[mask, :]^T @ r_)
                # That is, we use the full Z matrix to compute its covariance. However,
                # different number of samples are used, so we need to rescale.
                r_proj = Q_ @ (Q_.T @ r_) / (self.honest_splits_ratio * self.subsample)
                grad = r_ + (self.gamma - 1) * r_proj * dr_
                leaves_ = leaves[leaf_mask]
                ddr_ = ddr[leaf_mask]
            else:
                leaves_ = leaves
                dr_ = dr
                ddr_ = ddr
                Q_ = Q

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
            g = np.bincount(leaves_, weights=grad, minlength=num_leaves)

            # M^T diag(x) M = diag(np.bincount(leaves, weights=x))
            counts = np.bincount(
                leaves_,
                weights=dr_ + (self.gamma - 1) * r_proj * ddr_,
                minlength=num_leaves,
            )
            counts += self.params.get("lambda_l2", 0)

            # If honest_splits is True, it can occur that some leaves have no "leaf"
            # samples. We make the hessian invertible.
            counts[counts == 0] = 1

            # Mdr^T P_Z Mdr = (Mdr^T Q) @ (Mdr^T Q)^T
            # One could also compute this using bincount, but it appears this
            # version using a sparse matrix is faster.
            Mdr = scipy.sparse.csr_matrix(
                (
                    dr_,
                    (np.arange(len(leaves_)), leaves_),
                ),
                shape=(len(leaves_), num_leaves),
                dtype=dtype,
            )
            B = Mdr.T.dot(Q_)
            H = np.diag(counts) + (self.gamma - 1) * B @ B.T

            # Compute the 2nd order update
            leaf_values = -np.linalg.solve(H, g) * self.params.get("learning_rate", 0.1)

            for ldx, val in enumerate(leaf_values):
                self.booster.set_leaf_output(idx, ldx, val)

            # Ensure f == self.init_score_ + self.booster.predict(X)
            f += leaf_values[leaves]

        return self

    def predict(self, X, num_iteration=-1, raw_score=False):
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

        scores = self.booster.predict(X, num_iteration=num_iteration, raw_score=True)

        if self.objective == "logistic" and not raw_score:
            return scipy.special.expit(scores + self.init_score_)
        elif self.objective == "probit" and not raw_score:
            return scipy.stats.norm.cdf(scores + self.init_score_)
        else:
            return scores + self.init_score_
