# This test is a note to check I computed the gradient and hessian correctly.
import numpy as np
import pytest
import scipy
from scipy.optimize import approx_fprime



def test_probit_derivatives():
    def probit_loss(f, y):
        p = scipy.stats.norm.cdf(f)
        return np.where(y==1, -np.log(p), -np.log(1 - p))

    def probit_grad(f, y):
        p = scipy.stats.norm.cdf(f)
        dp = scipy.stats.norm.pdf(f)
        A = np.where(y == 1, - 1 / p, 1 / (1 - p))
        return A * dp

    def probit_hess(f, y):
        p = scipy.stats.norm.cdf(f)
        dp = scipy.stats.norm.pdf(f)
        A = np.where(y == 1, - 1 / p, 1 / (1 - p))
        return -f * dp * A + dp**2 * A**2

    def probit_hess2(f, y):
        p = scipy.stats.norm.cdf(f)
        dp = scipy.stats.norm.pdf(f)
        A = np.where(y == 1, -1 / p,- 1 / (p - 1))
        return (f**2 - 1) * dp * A - 3 * f * dp**2 * A**2 + 2 * dp**3 * A**3

    rtol = 1e-5
    for seed in range(10):
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, 0.5, 1)
        f = rng.normal(size=1)

        approx_grad = approx_fprime(f, lambda x: probit_loss(x, y))
        np.testing.assert_allclose(approx_grad, probit_grad(f, y), rtol)

        approx_hess = approx_fprime(f, lambda x: probit_grad(x, y))
        np.testing.assert_allclose(approx_hess, probit_hess(f, y), rtol)

        approx_hess2 = approx_fprime(f, lambda x: probit_hess(x, y))
        np.testing.assert_allclose(approx_hess2, probit_hess2(f, y), rtol)


def proj(A, f):
    return np.dot(A, np.linalg.lstsq(A, f, rcond=None)[0])

def proj_matrix(A):
    return np.dot(np.dot(A, np.linalg.inv(A.T @ A)), A.T)

def logistic_loss(X, beta, y, A, gamma):
    f = X @ beta
    p = scipy.special.expit(f)
    r = (y / 2 + 0.5) - p
    return -np.sum(np.log(np.where(y == 1, p, 1 - p))) + (gamma - 1) * np.sum(
        proj(A, r) ** 2
    )


def logistic_grad(X, beta, y, A, gamma):
    f = X @ beta
    p = scipy.special.expit(f)
    r = (y / 2 + 0.5) - p

    return (-r - 2 * (gamma - 1) * proj(A, r) * p * (1 - p)) @ X


def logistic_hess(X, beta, y, A, gamma):
    f = X @ beta
    p = scipy.special.expit(f)
    r = (y / 2 + 0.5) - p
    diag = +np.diag(p * (1 - p) * (1 - 2 * (gamma - 1) * (1 - 2 * p) * proj(A, r)))
    dense = proj_matrix(A) * (p * (1 - p))[np.newaxis, :] * (p * (1 - p))[:, np.newaxis]
    # dense = np.diag((p * (1 - p))) @ proj_matrix(A) @ np.diag((p * (1 - p)))
    return X.T @ (diag + 2 * (gamma - 1) * dense) @ X


def probit_loss(X, beta, y, A, gamma):
    f = X @ beta
    p = scipy.stats.norm.cdf(f)
    r = scipy.stats.norm.pdf(f) / np.where(y == 1, p, p - 1)
    return np.sum(-np.log(np.where(y == 1, p, 1 - p))) + (gamma - 1) * np.sum(
        proj(A, r) ** 2
    )


def probit_grad(X, beta, y, A, gamma):
    f = X @ beta
    p = scipy.stats.norm.cdf(f)
    pdf = scipy.stats.norm.pdf(f)
    denom = np.where(y == 1, p, p - 1)
    dl = -pdf / denom
    ddl = f * pdf / denom + pdf**2 / denom **2
    
    return (dl + 2 * (gamma - 1) * proj(A, dl) * ddl) @ X

def probit_hess(X, beta, y, Z, gamma):
    f = X @ beta

    p = scipy.stats.norm.cdf(f)
    dp = scipy.stats.norm.pdf(f)
    A = np.where(y == 1, -1 / p,- 1 / (p - 1))
    dl = A * dp
    ddl = -f * dp * A + dp**2 * A**2
    dddl = (f**2 - 1) * dp * A - 3 * f * dp**2 * A**2 + 2 * dp**3 * A**3
    
    a = X.T @ np.diag(ddl) @ X
    b = 2 * X.T @ np.diag(ddl) @ proj_matrix(Z) @ np.diag(ddl) @ X
    c = 2 * X.T @ np.diag(proj(Z, dl) * dddl) @ X
    return a + (gamma - 1) * (b + c)



@pytest.mark.parametrize("gamma", [1, 5])
@pytest.mark.parametrize(
    "loss, grad, hess",
    [
        (logistic_loss, logistic_grad, logistic_hess),
        (probit_loss, probit_grad, probit_hess),
    ],
)
def test_derivatives(loss, grad, hess, gamma, rtol=5e-3):
    rng = np.random.default_rng(0)
    n = 100
    p = 10
    q = 3

    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)

    y = 2 * rng.binomial(1, 0.5, n) - 1

    A = rng.normal(size=(n, q))

    approx_grad = approx_fprime(beta, lambda b: loss(X, b, y, A, gamma), 1e-5)
    np.testing.assert_allclose(approx_grad, grad(X, beta, y, A, gamma), rtol)

    approx_hess = approx_fprime(beta, lambda b: grad(X, b, y, A, gamma), 1e-4)
    np.testing.assert_allclose(approx_hess, hess(X, beta, y, A, gamma), rtol)
