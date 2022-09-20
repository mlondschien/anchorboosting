import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from anchorboost.anchor_loss import AnchorRegressionLoss

rng = np.random.default_rng(0)
n = 200
shift = 1.8

a = np.repeat([-1, 1, shift], n).reshape(-1, 1)
h = rng.normal(0, 1, size=(3 * n, 1))
x = a + h + rng.normal(0, 1, size=(3 * n, 1))
y = (x - a + 2 * h + rng.normal(0, 1, size=(3 * n, 1))).flatten()

gamma = 7
loss = AnchorRegressionLoss(gamma=gamma)
W = np.diag(np.ones(2 * n)) - (1 - np.sqrt(gamma)) * loss._proj_matrix(a[: (2 * n), :])

data = lgb.Dataset(x[: (2 * n), :], y[: (2 * n)])

data.anchor = a[: (2 * n), :]
a = a.flatten()

normal_lgb = lgb.train(
    params={"learning_rate": 0.1},
    train_set=data,
    num_boost_round=10,
)

anchor_lgb = lgb.train(
    params={"learning_rate": 0.1},
    train_set=data,
    num_boost_round=10,
    fobj=loss.objective,
)

normal_linear = LinearRegression().fit(x[: (2 * n), :], y[: (2 * n)])
anchor_linear = LinearRegression().fit(W @ x[: (2 * n), :], W @ y[: (2 * n)])

range_x = (x.min(), x.max())

fig = plt.figure(figsize=(12, 4))
grid = plt.GridSpec(2, 7)

# Scatter plot
scatterplot = fig.add_subplot(grid[:2, :3])
scatterplot.scatter(x[a == 1, :], y[a == 1], color="red", label="A = 1")
scatterplot.scatter(x[a == -1, :], y[a == -1], color="blue", label="A = -1")
scatterplot.scatter(
    x[a == shift, :], y[a == shift], color="green", label=f"A = {shift}"
)

linspace = np.linspace(*range_x, 1000).reshape(-1, 1)
scatterplot.plot(linspace, normal_lgb.predict(linspace), color="black", label="Normal")
scatterplot.plot(
    linspace,
    anchor_lgb.predict(linspace),
    color="blue",
    label=f"Anchor ({gamma})",
)
scatterplot.plot(
    linspace,
    normal_linear.predict(linspace),
    color="black",
)
scatterplot.plot(
    linspace,
    anchor_linear.predict(linspace),
    color="blue",
)

scatterplot.legend()

# LGBM Histograms
residuals_anchor = y - anchor_lgb.predict(x.reshape(-1, 1))
residuals_normal = y - normal_lgb.predict(x.reshape(-1, 1))

lgbm_histograms = [fig.add_subplot(grid[0, 3:5]), fig.add_subplot(grid[1, 3:5])]
lgbm_histograms[0].hist(residuals_normal[a == 1], bins=20, color="red", alpha=0.5)
lgbm_histograms[0].hist(residuals_normal[a == -1], bins=20, color="blue", alpha=0.5)
lgbm_histograms[0].hist(residuals_normal[a == shift], bins=20, color="green", alpha=0.5)

lgbm_histograms[0].vlines(residuals_normal[a == 1].mean(), 0, 50, color="red")
lgbm_histograms[0].vlines(residuals_normal[a == -1].mean(), 0, 50, color="blue")
lgbm_histograms[0].vlines(residuals_normal[a == shift].mean(), 0, 50, color="green")

# lgbm_histograms[0].set_xlabel("residuals")
lgbm_histograms[0].set_ylabel("count")
lgbm_histograms[0].set_title("GBT - Normal")

lgbm_histograms[1].hist(residuals_anchor[a == 1], bins=20, color="red", alpha=0.5)
lgbm_histograms[1].hist(residuals_anchor[a == -1], bins=20, color="blue", alpha=0.5)
lgbm_histograms[1].hist(residuals_anchor[a == shift], bins=20, color="green", alpha=0.5)

lgbm_histograms[1].vlines(residuals_anchor[a == 1].mean(), 0, 50, color="red")
lgbm_histograms[1].vlines(residuals_anchor[a == -1].mean(), 0, 50, color="blue")
lgbm_histograms[1].vlines(residuals_anchor[a == shift].mean(), 0, 50, color="green")

lgbm_histograms[1].set_xlabel("residuals")
lgbm_histograms[1].set_ylabel("count")
lgbm_histograms[1].set_title(f"GBT - Anchor ({gamma})")

# Linear Histograms
residuals_anchor = y - anchor_linear.predict(x.reshape(-1, 1))
residuals_normal = y - normal_linear.predict(x.reshape(-1, 1))

linear_histograms = [fig.add_subplot(grid[0, 5:7]), fig.add_subplot(grid[1, 5:7])]


linear_histograms[0].hist(residuals_normal[a == 1], bins=20, color="red", alpha=0.5)
linear_histograms[0].hist(residuals_normal[a == -1], bins=20, color="blue", alpha=0.5)
linear_histograms[0].hist(
    residuals_normal[a == shift], bins=20, color="green", alpha=0.5
)

linear_histograms[0].vlines(residuals_normal[a == 1].mean(), 0, 50, color="red")
linear_histograms[0].vlines(residuals_normal[a == -1].mean(), 0, 50, color="blue")
linear_histograms[0].vlines(residuals_normal[a == shift].mean(), 0, 50, color="green")

# linear_histograms[0].set_xlabel("residuals")
linear_histograms[0].set_ylabel("count")
linear_histograms[0].set_title("Linear - Normal")

linear_histograms[1].hist(residuals_anchor[a == 1], bins=20, color="red", alpha=0.5)
linear_histograms[1].hist(residuals_anchor[a == -1], bins=20, color="blue", alpha=0.5)
linear_histograms[1].hist(
    residuals_anchor[a == shift], bins=20, color="green", alpha=0.5
)

linear_histograms[1].vlines(residuals_anchor[a == 1].mean(), 0, 50, color="red")
linear_histograms[1].vlines(residuals_anchor[a == -1].mean(), 0, 50, color="blue")
linear_histograms[1].vlines(residuals_anchor[a == shift].mean(), 0, 50, color="green")

linear_histograms[1].set_xlabel("residuals")
linear_histograms[1].set_ylabel("count")
linear_histograms[1].set_title(f"Linear - Anchor ({gamma})")

plt.show()
