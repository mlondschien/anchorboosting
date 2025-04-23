import numpy as np
import pytest
from scipy.linalg import norm
from sklearn.preprocessing import OneHotEncoder

from anchorboosting.utils import proj

cases = [
    (
        np.array([[0], [0], [1]]),
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.5], [1.5], [3]]),
    ),
    (
        np.array([[2], [2], [1], [3]]),
        np.array([[1.0], [2.0], [3.0], [1.0]]),
        np.array([[1.5], [1.5], [3], [1]]),
    ),
    (
        np.array([[1], [1], [0]]),
        np.array([[1.0, 2.0], [0.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5, 2.0], [0.5, 2.0], [3.0, 4.0]]),
    ),
    (
        np.array([[1], [1], [0]]),
        np.array([1.0, 0.0, 3.0]),
        np.array([0.5, 0.5, 3]),
    ),
    (
        np.array([[1], [1], [0]]),
        np.array([[1.0, 2.0], [0.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5, 2.0], [0.5, 2.0], [3.0, 4.0]]),
    ),
    (
        np.array([[2], [2], [1]]),
        np.array([[1.0, 2.0], [0.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5, 2.0], [0.5, 2.0], [3.0, 4.0]]),
    ),
    (np.array([[0]]), np.array([[1]]), np.array([[1]])),
]


@pytest.mark.parametrize("anchor, residuals, result", cases)
@pytest.mark.parametrize("n_categories", [4, None])
def test_proj_result(n_categories, anchor, residuals, result):
    if n_categories is None:
        if anchor.shape[1] != 1:
            pytest.skip("one_hot only makes sense for 1D anchors")
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        anchor = encoder.fit_transform(anchor).astype(float)
    else:
        anchor = anchor.flatten()

    np.testing.assert_almost_equal(
        proj(anchor, residuals.copy(), n_categories=n_categories), result
    )


@pytest.mark.parametrize("anchor, residuals, _", cases)
def test_proj_dot_product(anchor, residuals, _):
    np.testing.assert_almost_equal(
        np.dot(proj(anchor, residuals).T, residuals),
        np.dot(proj(anchor, residuals).T, proj(anchor, residuals)),
    )


@pytest.mark.parametrize("anchor, residuals, result", cases)
@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_proj_orthogonal(anchor, residuals, result, gamma):
    np.testing.assert_almost_equal(
        norm(residuals - proj(anchor, residuals)) ** 2
        + gamma * norm(proj(anchor, residuals)) ** 2,
        norm(residuals) ** 2 + (gamma - 1) * norm(proj(anchor, residuals)) ** 2,
    )
