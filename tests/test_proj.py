import numpy as np
import pytest
import scipy.sparse

from anchorboosting.models import Proj

cases = [
    (
        np.array([1, 1, 3]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 1.5, 3.0]),
    ),
    (
        np.array([[0.0], [0.0], [1.0]]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 3.0]),
    ),
    (
        np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 1.5, 3]),
    ),
    (
        np.array([2, 2, 1, 7]),
        np.array([1.0, 2.0, 3.0, 1.0]),
        np.array([1.5, 1.5, 3.0, 1.0]),
    ),
    (
        np.array([1, 1, 0]),
        np.array([1.0, 0.0, 3.0]),
        np.array([0.5, 0.5, 3]),
    ),
    (np.array([0]), np.array([1.0]), np.array([1.0])),
    (np.array([[0.0]]), np.array([0.0]), np.array([0.0])),
]


@pytest.mark.parametrize("Z, f, result", cases)
def test_cached_proj_result(Z, f, result):
    np.testing.assert_almost_equal(Proj(Z)(f), result)


@pytest.mark.parametrize("Z, f, _", cases)
def test_cached_proj_dot_product(Z, f, _):
    np.testing.assert_almost_equal(
        np.dot(Proj(Z)(f).T, f),
        np.dot(Proj(Z)(f).T, Proj(Z)(f)),
    )


def test_proj_sandwich():
    n = 100
    num_categories = 10
    num_leaves = 5
    rng = np.random.RandomState(0)

    Z = rng.choice(np.arange(num_categories), size=n, replace=True)
    leaves = rng.choice(np.arange(num_leaves), size=n, replace=True)
    f = rng.normal(size=n)

    M = scipy.sparse.csr_matrix((f, (np.arange(n), leaves)), (n, num_leaves)).toarray()
    proj = Proj(Z)

    sandwich1 = proj.sandwich(leaves, num_leaves, f)
    for ldx in range(num_leaves):
        M[:, ldx] = proj(M[:, ldx])
    np.testing.assert_almost_equal(sandwich1, M.T @ M)
