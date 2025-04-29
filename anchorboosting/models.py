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
    honest_split_ratio: float, optional, default=0.5
        Ratio of training data used to determine tree splits. The rest is used to
        determine leaf values.
    **kwargs: dict
        Additional parameters for the LightGBM model. See LightGBM documentation for
        details.
    """

    def __init__(
        self,
        gamma,
        dataset_params=None,
        num_boost_round=100,
        honest_split_ratio=0.5,
        **kwargs,
    ):
        self.gamma = gamma
        self.params = kwargs
        self.dataset_params = dataset_params or {}
        self.num_boost_round = num_boost_round
        self.booster = None
        self.init_score_ = None
        self.honest_split_ratio = honest_split_ratio

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

        self.init_score_ = np.mean(y)

        dataset_params = {
            "data": X,
            "label": y,
            "categorical_feature": categorical_feature,
            "feature_name": feature_name,
            "init_score": np.ones(len(y)) * self.init_score_,
            **self.dataset_params,
        }

        data = lgb.Dataset(**dataset_params)
        data.anchor = Z

        self.booster = lgb.Booster(params=self.params, train_set=data)
        residuals = y - dataset_params["init_score"]

        Q, _ = np.linalg.qr(Z, mode="reduced")  # P_Z f = Q @ (Q^T @ f)

        rng = np.random.default_rng(self.params.get("random_state", 0))
        mask = np.empty_like(residuals, dtype=np.bool_)
        mask[: int(len(residuals) * self.honest_split_ratio)] = True
        mask[int(len(residuals) * self.honest_split_ratio) : ] = False

        for idx in range(self.num_boost_round):
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

            residuals_masked = residuals[mask]
            residuals_masked_proj = Q[mask, :] @ (Q[mask, :].T @ residuals_masked)
            grad = np.empty_like(residuals)
            grad[mask] = -residuals_masked - (self.gamma - 1) * residuals_masked_proj
            grad[~mask] = 0.0

            # is_finished is True if there we no splits satisfying the splitting
            # criteria. c.f. https://github.com/microsoft/LightGBM/pull/6890
            rng.shuffle(mask)
            grad_masked = np.where(mask, grad, 0.0)
            is_finished = self.booster._Booster__boost(
                grad_masked, mask.astype(np.float32)
            )

            if is_finished:
                print(f"Finished training after {idx} iterations.")
                break

            leaves = self.booster.predict(
                X, start_iteration=idx, num_iteration=1, pred_leaf=True
            ).flatten()
            num_leaves = np.max(leaves) + 1

            # Let M be the one-hot encoding of the tree's leaf assignments. That is,
            # M[i, j] = 1 if leaves[i] == j else 0. We wish to select the leaf values
            # beta to minimize the anchor loss:
            # loss = || y - M beta ||^2 + (gamma - 1) || P_Z (y - M beta) ||^2
            #      = || (Id + (gamma - 1) P_Z) (y - M beta) ||^2
            # The optimal beta is given by
            # beta = (M^T (Id + (gamma - 1) P_Z) M)^{-1} M^T (Id + (gamma - 1) P_Z) y
            # with M^T M = diag(np.bincount(leaves)).
            leaves_masked = leaves[~mask]
            M = scipy.sparse.csr_matrix(
                (
                    np.ones_like(leaves_masked),
                    (np.arange(len(leaves_masked)), leaves_masked),
                ),
                shape=(len(leaves_masked), num_leaves),
            )
            B = M.T.dot(Q[~mask, :])  # M^T @ Q of shape (num_leaves, num_anchors)
            # A = M^T (Id + (gamma - 1) P_Z) M, where M^T M = diag(np.bincount(leaves))
            counts = np.bincount(leaves_masked, minlength=num_leaves) * 1.0
            A = np.diag(counts) + (self.gamma - 1) * B @ B.T

            # b = M^T (Id + (gamma - 1) P_Z) y
            residuals_masked = residuals[~mask]
            residuals_masked_proj = Q[~mask, :] @ (Q[~mask, :].T @ residuals_masked)
            grad = -residuals_masked - (self.gamma - 1) * residuals_masked_proj
            b = np.bincount(leaves_masked, weights=grad, minlength=num_leaves)
            # b += (self.gamma - 1) * M.T.dot(residuals_proj)

            leaf_values = -np.linalg.solve(A, b) * self.params.get("learning_rate", 0.1)

            for ldx, val in enumerate(leaf_values):
                self.booster.set_leaf_output(idx, ldx, val)

            # Ensure residuals == y - self.init_score_ - self.booster.predict(X)
            residuals -= leaf_values[leaves]

        return self

    def predict(self, X, num_iteration=-1):
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
            X = X.to_arrow()

        scores = self.booster.predict(X, num_iteration=num_iteration, raw_score=True)
        return scores + self.init_score_
