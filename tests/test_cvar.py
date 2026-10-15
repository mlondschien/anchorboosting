import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime

from anchorboosting import AnchorBooster
from anchorboosting.models import Proj


def cvar_density(scores, weights, alpha):
    """Maximize sum(w*q*scores) with 0 <= q <= 1/alpha and sum(w*q)=1.

    Ties share a density. Weights must be positive and sum to one.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if alpha == 1:
        return np.ones_like(scores)
    # Group exact ties before allocating fractional mass at the tail boundary.
    _, inverse = np.unique(scores, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)[::-1]
    preceding = np.r_[0.0, np.cumsum(mass)[:-1]]
    density = np.clip((alpha - preceding) / mass, 0, 1) / alpha
    return density[::-1][inverse]


@pytest.mark.parametrize("categorical", [True, False])
def test_cvar_gradient_and_leaf_hessian(categorical):
    rng = np.random.default_rng(10)
    n = 70
    Z = rng.integers(0, 4, n) if categorical else rng.normal(size=(n, 3))
    proj = Proj(Z)
    r = rng.normal(size=n)
    gamma, alpha = 3, 0.35
    pr = proj(r)
    q = cvar_density(pr**2, np.full(n, 1 / n), alpha)
    grad = r - pr + gamma * proj(q * pr)
    # Numerical differentiation of the actual max-over-q loss, away from ties.

    def loss(v):
        projected = proj(v)
        q = cvar_density(projected**2, np.full(n, 1 / n), alpha)
        return (
            (v - projected) @ (v - projected) + gamma * (q * projected) @ projected
        ) / 2

    numerical = approx_fprime(r, loss, 1e-7)
    assert_allclose(grad, numerical, atol=3e-6)
    leaves = rng.integers(0, 5, n)
    M = np.eye(5)[leaves]
    P = np.column_stack([proj(col) for col in np.eye(n)])
    H = np.diag(np.bincount(leaves)) - proj.sandwich(leaves, 5, np.ones(n))
    H += gamma * proj.sandwich(leaves, 5, np.ones(n), projection_weights=q)
    expected = M.T @ (np.eye(n) - P + gamma * P @ np.diag(q) @ P) @ M
    assert_allclose(H, expected, atol=1e-12)
    hessian_fd = approx_fprime(
        np.zeros(5),
        lambda b: M.T
        @ ((r + M @ b) - proj(r + M @ b) + gamma * proj(q * proj(r + M @ b))),
        1e-6,
    )
    assert_allclose(H, hessian_fd, atol=1e-7)


def sample():
    rng = np.random.default_rng(42)
    Z = rng.integers(0, 4, 250)
    X = rng.normal(size=(250, 3)) + Z[:, None] / 3
    y = X[:, 0] ** 2 + X[:, 1] + Z + rng.normal(size=250)
    return X, y, Z


@pytest.mark.parametrize("gamma", [0, 0.25, 1, 3])
@pytest.mark.parametrize("alpha", [1, 0.3, 0.001])
@pytest.mark.parametrize("categorical", [True, False])
def test_boosting_recomputes_densities_and_uses_leaf_hessian(gamma, alpha, categorical):
    X, y, Z = sample()
    if not categorical:
        Z = np.column_stack([np.ones(len(y)), X])
    rate = 0.05
    model = AnchorBooster(
        gamma=gamma,
        alpha_cvar=alpha,
        num_boost_round=8,
        learning_rate=rate,
        num_leaves=4,
        min_gain_to_split=0.01,
        num_threads=1,
        verbosity=-1,
    ).fit(X, y, Z)
    proj = Proj(Z)
    f = np.full(len(y), model.init_score_)
    assert model.booster_.num_trees() == 8
    densities = []
    for idx in range(model.booster_.num_trees()):
        pr = proj(f - y)
        q_obs = cvar_density(pr**2, np.full(len(y), 1 / len(y)), alpha)
        densities.append(q_obs)
        leaves = model.booster_.predict(
            X, pred_leaf=True, start_iteration=idx, num_iteration=1
        ).ravel()
        n_leaves = leaves.max() + 1
        grad = f - y - pr + gamma * proj(q_obs * pr)
        H = np.diag(np.bincount(leaves)).astype(float)
        H -= proj.sandwich(leaves, n_leaves, np.ones(len(y)))
        H += gamma * proj.sandwich(leaves, n_leaves, np.ones(len(y)), q_obs)
        expected = (
            -rate
            * np.linalg.lstsq(H, np.bincount(leaves, weights=grad), rcond=1e-12)[0]
        )
        actual = np.array(
            [model.booster_.get_leaf_output(idx, leaf) for leaf in range(n_leaves)]
        )
        assert_allclose(actual, expected, atol=1e-9)
        f += actual[leaves]
    assert_allclose(model.predict(X), f)
    densities = np.array(densities)
    assert_allclose(densities.mean(axis=1), 1)
    assert np.all(densities >= 0)
    assert np.all(densities <= 1 / alpha)
    if gamma == 1 and alpha < 1:
        assert np.any(np.ptp(densities, axis=0) > 0)


def test_categorical_matches_onehot_and_label_permutation():
    X, y, Z = sample()
    kwargs = dict(
        gamma=2,
        alpha_cvar=0.4,
        num_boost_round=5,
        num_leaves=4,
        min_gain_to_split=0.01,
        num_threads=1,
        verbosity=-1,
    )
    a = AnchorBooster(**kwargs).fit(X, y, Z)
    b = AnchorBooster(**kwargs).fit(X, y, np.eye(4)[Z])
    c = AnchorBooster(**kwargs).fit(X, y, np.array([50, 3, 200, 8])[Z])
    assert_allclose(a.predict(X), b.predict(X), atol=1e-10)
    assert_allclose(a.predict(X), c.predict(X), atol=1e-10)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    X, y, Z = sample()
    with pytest.raises(ValueError, match="alpha_cvar"):
        AnchorBooster(gamma=1, alpha_cvar=alpha).fit(X, y, Z)


def test_unsupported_inputs():
    X, y, Z = sample()
    with pytest.raises(ValueError, match="regression with supplied anchors"):
        AnchorBooster(gamma=1, alpha_cvar=0.1).fit(X, y)
    with pytest.raises(ValueError, match="regression with supplied anchors"):
        AnchorBooster(gamma=1, alpha_cvar=0.1, objective="binary").fit(X, y > 0, Z)
