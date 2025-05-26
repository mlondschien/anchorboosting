import numpy as np
import polars as pl


def simulate(f, n=100, shift=0, seed=0):
    rng = np.random.RandomState(seed)

    p = 3

    a = rng.normal(size=(n, 2)) + shift
    h = rng.normal(size=(n, 1))
    x_noise = 0.5 * rng.normal(size=(n, p))
    x = x_noise + np.repeat(a[:, 0] + a[:, 1] + 2 * h[:, 0], p).reshape((n, p))

    y_noise = 0.25 * rng.normal(size=n)
    y = f(x[:, 1], x[:, 2]) - 2 * a[:, 0] + 3 * h[:, 0] + y_noise

    quantiles = np.quantile(x[:, 2], q=np.linspace(0, 1, 11))
    x3 = np.digitize(x[:, 2], bins=quantiles)

    df = pl.DataFrame(
        {"x1": x[:, 0], "x2": x[:, 1], "x3": x3},
        schema={"x1": pl.Float64, "x2": pl.Float64, "x3": pl.Int64},
    )

    return df, y, a


def f1(x2, x3):
    return (x2 <= 0) + (x2 <= -0.5) * (x3 <= 1)


def f2(x2, x3):
    return x2 + x3 + (x2 <= 0) + (x2 <= -0.5) * (x3 <= 1)
