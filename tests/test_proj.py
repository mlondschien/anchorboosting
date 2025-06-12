import numpy as np
import pytest

from anchorboosting.models import cached_proj

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
    np.testing.assert_almost_equal(cached_proj(Z)(f), result)


@pytest.mark.parametrize("Z, f, _", cases)
def test_cached_proj_dot_product(Z, f, _):
    np.testing.assert_almost_equal(
        np.dot(cached_proj(Z)(f).T, f),
        np.dot(cached_proj(Z)(f).T, cached_proj(Z)(f)),
    )
