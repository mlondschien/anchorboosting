import numpy as np
import pytest

# --- Functions to test ---
def simulate(f, n=100, shift=0, seed=0):
    rng = np.random.RandomState(seed)
    p = 2
    a = rng.normal(size=(n, 2)) + shift
    h = rng.normal(size=(n, 1))
    x_noise = 0.5 * rng.normal(size=(n, p))
    x = x_noise + np.repeat(a[:, 0] + a[:, 1] + 2 * h[:, 0], p).reshape((n, p))
    y_noise = 0.25 * rng.normal(size=n)
    y = f(x[:, 0], x[:, 1]) - 2 * a[:, 0] + 3 * h[:, 0] + y_noise
    return x, y, a

def f1(x2, x3):
    return (x2 <= 0) + (x2 <= -0.5) * (x3 <= 1)

def f2(x2, x3):
    return x2 + x3 + (x2 <= 0) + (x2 <= -0.5) * (x3 <= 1)

# --- Test ---
@pytest.mark.parametrize("f", [f1, f2])
def test_simulate_output(f):
    n = 100
    x, y, a = simulate(f, n=n, shift=2, seed=123)
    assert x.shape == (n, 2)
    assert y.shape == (n,)
    assert a.shape == (n, 2)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(a))
    x2, y2, a2 = simulate(f, n=n, shift=2, seed=123)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(a, a2)