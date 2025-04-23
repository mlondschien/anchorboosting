# copy-pasted from https://github.com/mlondschien/ivmodels/blob/09f8bc5a78b3793ae5\
# 1f0bb7f0331b70a0971ebc/ivmodels/utils.py
import numpy as np
import scipy


def proj(Z, f, n_categories=None, copy=False):
    """
    Project f onto the subspace spanned by Z.

    If `Z` is a matrix of floats, uses a copy of `ivmodel`'s `proj:
    https://github.com/mlondschien/ivmodels/blob/main/ivmodels/utils.py#L12
    If `Z` is a 1d array of unsigned integers, assumes that `Z` is categorical with
    values 0, 1, ..., n_categories-1. In this case, the `n_categories` argument must be
    be supplied. The projection is done by averaging the values of f within each
    category. If `Z` is None, returns np.zeros_like(f).

    Parameters
    ----------
    Z: np.ndarray of dimension (n, d_Z) or (n,)
        The `Z` matrix or 1d array of integers. If None, returns np.zeros_like(f).
    *args: np.ndarrays of dimension (n, d_f) or (n,)
        Vectors or matrices to project.
    n_categories: int, optional, default=None
        If not None, then Z is assumed to be categorical with categories 0, 1, ...,
        n_categories-1. The projection is done by averaging the values of f within each
        category.
    copy: bool, default=False
        If False, overwrites *args.

    Returns
    -------
    np.ndarray of dimension (n, d_f) or (n,)
        Projection of args onto the subspace spanned by Z. Same number of
        outputs as args. Same dimension as args
    """
    if Z is None:
        return np.zeros_like(f)

    if len(f.shape) > 2:
        raise ValueError(f"*args should have shapes (n, d_f) or (n,). Got {f.shape}.")
    if f.shape[0] != Z.shape[0]:
        raise ValueError(f"Shape mismatch: Z.shape={Z.shape}, f.shape={f.shape}.")

    if n_categories is not None:
        f = f.copy() if copy else f

        if len(Z.shape) != 1 or not np.issubdtype(Z.dtype, np.integer):
            raise ValueError(
                "If n_categories is not None, then Z should be a single column of "
                f"integers. Got shape {Z.shape} and dtype {Z.dtype}."
            )

        if len(f.shape) == 1:
            means = np.zeros((n_categories,))
            counts = np.zeros(n_categories)

            # np.add.at(a, indices, b) is equivalent to a[indices] += b
            np.add.at(means, Z, f)
            np.add.at(counts, Z, 1)

            means[counts > 0] = means[counts > 0] / counts[counts > 0]

            f[:] = means[Z]
        else:
            means = np.zeros((n_categories, f.shape[1]))
            counts = np.zeros(n_categories)

            # np.add.at(a, indices, b) is equivalent to a[indices] += b
            np.add.at(means, Z, f)
            np.add.at(counts, Z, 1)

            means[counts > 0, :] = means[counts > 0, :] / counts[counts > 0, None]

            f[:] = means[Z, :]

        return f

    # The gelsy driver raises in this case - we handle it separately
    if len(f.shape) == 2 and f.shape[1] == 0:
        return np.zeros_like(f)

    # return np.dot(Z, scipy.linalg.pinv(Z.T @ Z) @ Z.T @ args[0])
    return np.dot(Z, scipy.linalg.lstsq(Z, f, cond=None, lapack_driver="gelsy")[0])
