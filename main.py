import lightgbm as lgb

from anchorboost.anchor_loss import AnchorL2Loss
from anchorboost.simulate import f1, simulate

loss = AnchorL2Loss(10)

X_fit, y_fit, a_fit = simulate(f1, shift=0, seed=0)
X_val, y_val, a_val = simulate(f1, shift=0, seed=1)
X_ood, y_ood, a_ood = simulate(f1, shift=3, seed=2)

fit = lgb.Dataset(X_fit, y_fit)
fit.anchor = a_fit

val = lgb.Dataset(X_val, y_val, reference=fit)
val.anchor = a_val

ood = lgb.Dataset(X_ood, y_ood, reference=fit)
ood.anchor = a_ood

anchor_loss = AnchorL2Loss(10)

model = lgb.train(
    params={"learning_rate": 0.01},
    train_set=fit,
    num_boost_round=100,
    valid_sets=(fit, val, ood),
    valid_names=("fit", "val", "ood"),
    fobj=anchor_loss.lgb_anchor_objective,
    feval=(anchor_loss.lgb_anchor_score, anchor_loss.lgb_l2_score),
)
