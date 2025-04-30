# This test is a note to check I computed the gradient and hessian correctly.
import numpy as np
import pytest
from scipy.optimize import approx_fprime


def predictions(f):
    return 1 / (1 + np.exp(-f))


def proj(A, f):
    return np.dot(A, np.linalg.lstsq(A, f, rcond=None)[0])


def proj_matrix(A):
    return np.dot(np.dot(A, np.linalg.inv(A.T @ A)), A.T)


def loss(X, beta, y, A, gamma):
    f = X @ beta
    p = predictions(f)
    r = (y / 2 + 0.5) - p  # for y in {-1, 1} -> {0, 1}
    return -np.sum(np.log(np.where(y == 1, p, 1 - p))) + (gamma - 1) * np.sum(
        proj(A, r) ** 2
    )


def grad(X, beta, y, A, gamma):
    f = X @ beta
    p = predictions(f)
    r = (y / 2 + 0.5) - p

    return (-r - 2 * (gamma - 1) * proj(A, r) * p * (1 - p)) @ X


def hess(X, beta, y, A, gamma):
    f = X @ beta
    p = predictions(f)
    r = (y / 2 + 0.5) - p
    diag = +np.diag(p * (1 - p) * (1 - 2 * (gamma - 1) * (1 - 2 * p) * proj(A, r)))
    dense = proj_matrix(A) * (p * (1 - p))[np.newaxis, :] * (p * (1 - p))[:, np.newaxis]
    # dense = np.diag((p * (1 - p))) @ proj_matrix(A) @ np.diag((p * (1 - p)))
    return X.T @ (diag + 2 * (gamma - 1) * dense) @ X


@pytest.mark.parametrize("gamma", [0, 0.1, 0.99, 1, 5, 50])
def test_grad_hess(gamma):
    rng = np.random.default_rng(0)
    n = 100
    p = 10
    q = 3

    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)

    y = 2 * rng.binomial(1, 0.5, n) - 1

    A = rng.normal(size=(n, q))

    approx_grad = approx_fprime(beta, lambda b: loss(X, b, y, A, gamma))
    np.testing.assert_allclose(approx_grad, grad(X, beta, y, A, gamma), 1e-5)

    approx_hess = approx_fprime(beta, lambda b: grad(X, b, y, A, gamma), 1e-7)
    np.testing.assert_allclose(approx_hess, hess(X, beta, y, A, gamma), 1e-5)
