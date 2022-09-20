import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from anchorboost.anchor_loss import AnchorRegressionLoss
from anchorboost.simulate import f1, simulate

X_fit, y_fit, a_fit = simulate(f1, shift=0, seed=0)
X_val, y_val, a_val = simulate(f1, shift=0, seed=1)
X_ood, y_ood, a_ood = simulate(f1, shift=2, seed=2)

fit = lgb.Dataset(X_fit, y_fit)
fit.anchor = a_fit

val = lgb.Dataset(X_val, y_val, reference=fit)
val.anchor = a_val

ood = lgb.Dataset(X_ood, y_ood, reference=fit)
ood.anchor = a_ood

l2_loss = AnchorRegressionLoss(1)
anchor_loss = AnchorRegressionLoss(7)

print(
    f"Mean model: fit: {np.mean((y_fit - np.mean(y_fit))**2)}, \
    val: {np.mean((y_val - np.mean(y_fit))**2)}, \
    ood: {np.mean((y_ood - np.mean(y_fit))**2)}"
)

print("\n training lgbm model")
model = lgb.train(
    params={"learning_rate": 0.05},
    train_set=fit,
    num_boost_round=10,
    valid_sets=(fit, val, ood),
    verbose_eval=1,
    valid_names=("fit", "val", "ood"),
    feval=(anchor_loss.score),
)

print("\n training l2 model")
lgbm_model = lgb.train(
    params={"learning_rate": 0.05},
    train_set=fit,
    num_boost_round=10,
    valid_sets=(fit, val, ood),
    valid_names=("fit", "val", "ood"),
    fobj=l2_loss.objective,
    verbose_eval=1,
    feval=(l2_loss.score, anchor_loss.score),
)

print("\n training anchor model")
model = lgb.train(
    params={"learning_rate": 0.05},
    train_set=fit,
    num_boost_round=10,
    valid_sets=(fit, val, ood),
    valid_names=("fit", "val", "ood"),
    verbose_eval=1,
    fobj=anchor_loss.objective,
    feval=(l2_loss.score, anchor_loss.score),
)


# fig, axes = plt.subplots(1, 1, figsize=(15, 5))#, projection="3d")
fig = plt.figure()
axes = plt.axes(projection="3d")
Xp, Yp = np.meshgrid(
    np.linspace(
        min(X_fit[:, 0].min(), X_ood[:, 0].min()),
        max(X_fit[:, 0].max(), X_ood[:, 0].max()),
        100,
    ),
    np.linspace(
        min(X_fit[:, 1].min(), X_ood[:, 1].min()),
        max(X_fit[:, 1].max(), X_ood[:, 1].max()),
        100,
    ),
)
Zp_lgbm = lgbm_model.predict(
    np.concatenate([Xp.flatten()[:, None], Yp.flatten()[:, None]], axis=1)
)
axes.plot_surface(Xp, Yp, Zp_lgbm.reshape(Xp.shape), rstride=1, cstride=1)
axes.scatter(X_ood[:, 0], X_ood[:, 1], y_ood, c="r", marker="x")
axes.scatter(X_fit[:, 0], X_fit[:, 1], y_ood, c="b", marker="x")
plt.show()
