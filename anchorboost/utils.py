# copy-pasted from https://github.com/mlondschien/ivmodels/blob/09f8bc5a78b3793ae5\
# 1f0bb7f0331b70a0971ebc/ivmodels/utils.py
import numpy as np
import scipy

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    _PANDAS_INSTALLED = False

try:
    import polars as pl

    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False


def proj(Z, *args, categorical_Z=False):
    """Project f onto the subspace spanned by Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix. If None, returns np.zeros_like(f).
    *args: np.ndarrays of dimension (n, d_f) or (n,)
        vector or matrices to project.
    categorical_Z: bool, default=False
        If True, then Z is assumed to be categorical and the projection is done by
        averaging the values of f within each category.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of args onto the subspace spanned by Z. Same number of
        outputs as args. Same dimension as args
    """
    if Z is None:
        return (*(np.zeros_like(f) for f in args),)

    if categorical_Z:
        if (len(Z.shape) != 1 and Z.shape[1] != 1) or "float" in str(Z.dtype):
            raise ValueError(
                "If categorical_Z=True, then Z should be a single column of integers "
                f"or string. Got shape {Z.shape} and dtype {Z.dtype}."
            )
        out = [np.zeros_like(a, dtype="float") for a in args]
        for unique_value in np.unique(Z):
            mask = (Z == unique_value).flatten()
            for i, f in enumerate(args):
                if len(f.shape) == 1:
                    out[i][mask] = f[mask].mean()
                else:
                    out[i][mask, :] = f[mask, :].mean(axis=0)

        if len(args) == 1:
            return out[0]
        return tuple(out)

    for f in args:
        if len(f.shape) > 2:
            raise ValueError(
                f"*args should have shapes (n, d_f) or (n,). Got {f.shape}."
            )
        if f.shape[0] != Z.shape[0]:
            raise ValueError(f"Shape mismatch: Z.shape={Z.shape}, f.shape={f.shape}.")

    if len(args) == 1:
        # The gelsy driver raises in this case - we handle it separately
        if len(args[0].shape) == 2 and args[0].shape[1] == 0:
            return np.zeros_like(args[0])

        # return np.dot(Z, scipy.linalg.pinv(Z.T @ Z) @ Z.T @ args[0])
        return np.dot(
            Z, scipy.linalg.lstsq(Z, args[0], cond=None, lapack_driver="gelsy")[0]
        )

    csum = np.cumsum([f.shape[1] if len(f.shape) == 2 else 1 for f in args])
    csum = [0] + csum.tolist()

    fs = np.hstack([f.reshape(Z.shape[0], -1) for f in args])

    if fs.shape[1] == 0:
        # The gelsy driver raises in this case - we handle it separately
        return (*(np.zeros_like(f) for f in args),)

    # fs = np.dot(Z, scipy.linalg.pinv(Z.T @ Z) @ Z.T @ fs)
    fs = np.dot(Z, scipy.linalg.lstsq(Z, fs, cond=None, lapack_driver="gelsy")[0])
    return (
        *(fs[:, i:j].reshape(f.shape) for i, j, f in zip(csum[:-1], csum[1:], args)),
    )


def oproj(Z, *args, categorical_Z=False):
    """Project f onto the subspace orthogonal to Z.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z)
        The Z matrix. If None, returns f.
    *args: np.ndarrays of dimension (n, d_f) or (n,)
        vector or matrices to project.
    allow_categorical_Z: bool, default=True
        If True, and Z.shape[1] == 1 and Z.dtype is int, then Z is assumed to be
        categorical and the projection is done by averaging the values of f within each
        category.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of args onto the subspace spanned by Z. Same number of
        outputs as args. Same dimension as args
    """
    if Z is None:
        return (*args,)

    if len(args) == 1:
        return args[0] - proj(Z, args[0], categorical_Z=categorical_Z)

    else:
        return (
            *(
                x - x_proj
                for x, x_proj in zip(args, proj(Z, *args, categorical_Z=categorical_Z))
            ),
        )


def to_numpy(*args):
    """Convert input args to a numpy array."""
    out = []
    for x in args:
        if x is None:
            out.append(None)
        elif isinstance(x, np.ndarray):
            out.append(x)
        elif _PANDAS_INSTALLED and isinstance(x, (pd.DataFrame, pd.Series)):
            out.append(x.to_numpy())
        elif _POLARS_INSTALLED and isinstance(x, pl.DataFrame):
            out.append(x.to_numpy())
        else:
            raise ValueError(f"Invalid type: {type(x)}")

    if len(args) == 1:
        return out[0]
    else:
        return (*out,)
