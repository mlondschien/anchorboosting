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
        The objective function to use. Can be "regression" or "binary".
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
            categorical_feature = (
                [X.columns.index(col) for col in categorical_feature]
                if categorical_feature
                else None
            )
            X = X.to_arrow()
        else:
            feature_name = None

        dtype = np.result_type(Z, y)

        self.init_score_ = np.mean(y)
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
        hess = np.ones(len(y), dtype=dtype)
        for idx in range(self.num_boost_round):
            # For regression, the gradient is grad = (y - f) + (gamma - 1) P_Z (y - f)
            if self.objective == "regression":
                residuals = y - f

            # For classification, predictions are p = 1 / (1 + exp(-f) and the gradient
            # grad = (y - p) - 2 * (gamma - 1) P_Z (y - p) * p * (1 - p)
            else:
                p = 1 / (1 + np.exp(-f))
                px1mp = p * (1 - p)
                residuals = y - p

            residuals_proj = Q @ (Q.T @ residuals)

            if self.objective == "regression":
                grad = -residuals - (self.gamma - 1) * residuals_proj
            else:
                grad = -residuals - 2 * (self.gamma - 1) * residuals_proj * px1mp

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
            is_finished = self.booster._Booster__boost(grad, hess)

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

            # Regression
            # The anchor regression objective is
            # L = || y - f ||^2 + (gamma - 1) || P_Z (y - f) ||^2
            # The gradient of the anchor regression objective w.r.t. f is
            # g = - (y - f) - (gamma - 1) P_Z (y - f) = - (Id + (gamma - 1)) P_Z (y - f)
            # The hessian of the anchor regression objective w.r.t. f is
            # H = Id + (gamma - 1) P_Z
            # The gradient of the anchor regression objective w.r.t. the leaf values is
            # g = - M^T (Id + (gamma - 1) P_Z) (y - f)
            # The hessian of the anchor regression objective w.r.t. the leaf values is
            # H = M^T (Id + (gamma - 1) P_Z) M
            # We do the 2nd order update
            # beta = (M^T (Id + (gamma-1) P_Z) M)^{-1} M^T (Id + (gamma-1) P_Z) (y - f)
            # As L is quadratic in f, this is the global minimizer of L w.r.t. f.
            if self.objective == "regression":
                # M^T grad = bincount(leaves_masked, grad)
                g = np.bincount(leaves, weights=grad, minlength=num_leaves)

                # M^T M = diag(np.bincount(leaves))
                counts = np.bincount(leaves, minlength=num_leaves)

                # M^T P_Z M = (M^T Q) @ (M^T Q)^T
                # One could also compute this using bincount, but it appears this
                # version using a sparse matrix is faster.
                M = scipy.sparse.csr_matrix(
                    (
                        np.ones_like(leaves),
                        (np.arange(len(leaves)), leaves),
                    ),
                    shape=(len(leaves), num_leaves),
                    dtype=dtype,
                )
                B = M.T.dot(Q)
                H = np.diag(counts) + (self.gamma - 1) * B @ B.T

            # Classification:
            # The anchor classification objective (loss) is
            # L = - sum_i (y_i log(p_i) + (1 - y_i) log(1 - p_i))
            #                                     + (gamma - 1) || P_Z (y - p) ||^2
            # The gradient of the anchor classification objective w.r.t. f is
            # g = - (y - p) - 2 * (gamma - 1) P_Z (y - p) * p * (1 - p)
            # The hessian of the anchor classification objective w.r.t. f is
            # H = diag[p * (1 - p) * {1 - 2 * (gamma - 1) * (1 - 2 * p) P_Z (y - p)}]
            #     + 2 * (gamma - 1) * diag(p * (1 - p)) * P_Z * diag(p * (1 - p))
            # The gradient of the anchor classification objective w.r.t. the leaf values
            # is
            # g = - M^T [ (y - p) - 2 * (gamma - 1) P_Z (y - p) * p * (1 - p) ]
            # The hessian of the anchor classification objective w.r.t. the leaf values
            # is
            # H = M^T diag[p * (1 - p) *
            #                        {1 - 2 * (gamma - 1) * (1 - 2 * p) P_Z (y - p)}] M
            #   + 2 * (gamma - 1) M^T diag(p * (1 - p)) P_Z diag(p * (1 - p)) M
            else:
                # M^T grad = bincount(leaves, grad)
                g = np.bincount(leaves, weights=grad, minlength=num_leaves)

                weights = px1mp * (
                    1 - 2 * (self.gamma - 1) * (1 - 2 * p) * residuals_proj
                )
                counts = np.bincount(leaves, weights=weights, minlength=num_leaves)

                # We directly compute M <- diag(p * (1 - p)) M
                M = scipy.sparse.csr_matrix(
                    (
                        px1mp,
                        (np.arange(len(leaves)), leaves),
                    ),
                    shape=(len(leaves), num_leaves),
                    dtype=dtype,
                )
                B = M.T.dot(Q)  # M^T @ Q of shape (num_leaves, num_anchors)
                H = np.diag(counts) + 2 * (self.gamma - 1) * B @ B.T

            # Compute the 2nd order update
            leaf_values = -np.linalg.solve(H, g) * self.params.get("learning_rate", 0.1)

            for ldx, val in enumerate(leaf_values):
                self.booster.set_leaf_output(idx, ldx, val)

            # Ensure residuals == y - self.init_score_ - self.booster.predict(X)
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
        """
        if self.booster is None:
            raise ValueError("AnchorBoost has not yet been fitted.")

        if _POLARS_INSTALLED and isinstance(X, pl.DataFrame):
            X = X.numpy()

        scores = self.booster.predict(X, num_iteration=num_iteration, raw_score=True)

        if self.objective in ["classification", "binary"] and not raw_score:
            return 1 / (1 + np.exp(-scores - self.init_score_))
        else:
            return scores + self.init_score_
