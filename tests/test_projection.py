import numpy as np
import pytest
from scipy.linalg import norm

from anchorboost.anchor_loss import AnchorMixin

cases = [
    (
        np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0]]),
        np.array([[1], [2], [3]]),
        np.array([[1.5], [1.5], [3]]),
    ),
    (
        np.array([[2], [2], [1], [3]]),
        np.array([[1], [2], [3], [1]]),
        np.array([[1.5], [1.5], [3], [1]]),
    ),
    (
        np.array([[1.0], [1.0], [0.0]]),
        np.array([[1, 2], [0, 2], [3, 4]]),
        np.array([[0.5, 2.0], [0.5, 2.0], [0.0, 0.0]]),
    ),
    (
        np.array([[2], [2], [1]]),
        np.array([[1, 2], [0, 2], [3, 4]]),
        np.array([[0.5, 2.0], [0.5, 2.0], [3.0, 4.0]]),
    ),
    (np.array([[0.0]]), np.array([[1]]), np.array([[0]])),
]


@pytest.mark.parametrize("anchor, residuals, result", cases)
def test_proj_result(anchor, residuals, result):
    loss = AnchorMixin(1)
    np.testing.assert_almost_equal(loss._proj(anchor, residuals), result)


@pytest.mark.parametrize("anchor, residuals, _", cases)
def test_proj_dot_product(anchor, residuals, _):
    loss = AnchorMixin(1)

    np.testing.assert_almost_equal(
        np.dot(loss._proj(anchor, residuals).T, residuals),
        np.dot(loss._proj(anchor, residuals).T, loss._proj(anchor, residuals)),
    )


@pytest.mark.parametrize("anchor, residuals, result", cases)
@pytest.mark.parametrize("gamma", [0.1, 1, 2, 10, 100, 1000])
def test_proj_orthogonal(anchor, residuals, result, gamma):
    loss = AnchorMixin(1)

    np.testing.assert_almost_equal(
        norm(residuals - loss._proj(anchor, residuals)) ** 2
        + gamma * norm(loss._proj(anchor, residuals)) ** 2,
        norm(residuals) ** 2 + (gamma - 1) * norm(loss._proj(anchor, residuals)) ** 2,
    )
