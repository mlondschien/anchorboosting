from functools import partial

import lightgbm as lgb
import numpy as np
import scipy

from anchorboosting.utils import proj

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
    **kwargs: dict
        Additional parameters for the LightGBM model. See LightGBM documentation for
        details.
    """

    def __init__(
        self,
        gamma,
        dataset_params=None,
        num_boost_round=100,
        **kwargs,
    ):
        self.gamma = gamma
        self.params = kwargs
        self.dataset_params = dataset_params or {}
        self.num_boost_round = num_boost_round
        self.booster = None
        self.init_score_ = None

    def fit(
        self,
        X,
        y,
        Z=None,
        n_categories=None,
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
            Anchor. Matrix of floats or 1d array of integers 0, ..., n_categories - 1.
        n_categories : int
            If anchor is a 1d array of integers, this is the number of categories.
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
        mult = np.sqrt(self.gamma) - 1
        M = np.empty((len(y), self.params.get("num_leaves", 31) + 1), dtype=np.float64)
        residuals = y - dataset_params["init_score"]

        proj_Z = _get_proj(Z=Z, n_categories=n_categories)
        hess = np.ones(len(y), dtype=np.float64)

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

            residuals_proj = proj_Z(residuals, copy=True)
            grad = -residuals - (self.gamma - 1) * residuals_proj
            # is_finished is True if there we no splits satisfying the splitting
            # criteria. c.f. https://github.com/microsoft/LightGBM/pull/6890
            is_finished = self.booster._Booster__boost(grad, hess)

            if is_finished:
                print(f"Finished training after {idx} iterations.")
                break

            leaves = self.booster.predict(
                X, start_iteration=idx, num_iteration=1, pred_leaf=True
            )
            num_leaves = np.max(leaves) + 1

            M[:, :num_leaves] = np.equal.outer(leaves, np.arange(num_leaves))
            M[:, :num_leaves] += mult * proj_Z(M[:, :num_leaves], copy=False)
            residuals_mult = residuals + mult * residuals_proj

            leaf_values = (
                self.params.get("learning_rate", 0.1) * scipy.linalg.lstsq(
                    M[:, :num_leaves], residuals_mult, cond=None, lapack_driver="gelsy"
                )[0]
            )

            for ldx, val in enumerate(leaf_values):
                self.booster.set_leaf_output(idx, ldx, val)
                # Ensure residuals == y - self.init_score_ - self.booster.predict(X)
                residuals[leaves == ldx] -= val

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


def _get_proj(Z, n_categories=None):
    if n_categories is not None:
        return partial(proj, Z=Z, n_categories=n_categories)
    else:
        pinvZ = np.linalg.pinv(Z)

        def proj_precomputed(*args, copy=False):
            if len(args) == 1:
                return np.dot(Z, pinvZ @ args[0])
            else:
                return (*(np.dot(Z, pinvZ @ f) for f in args),)

        return proj_precomputed
