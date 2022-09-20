import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anchorboost.anchor_loss import AnchorClassificationLoss

df = pd.read_csv("data/France_crop_type_data.csv").iloc[:, 1:]


def one_hot(array, center=False):
    unique_values, unique_indices = np.unique(array, return_inverse=True)
    one_hot = np.zeros((len(array), len(unique_values)))
    one_hot[np.arange(len(array)), unique_indices] = 1

    if center:
        one_hot -= one_hot.mean(axis=0)

    return one_hot


def train_test_ood_split(X, y, e, ood_environment, test_size=0.2, seed=0):
    in_distribution = e != ood_environment
    X_train_test = X[in_distribution]
    y_train_test = y[in_distribution]
    e_train_test = e[in_distribution].reshape(-1, 1)  # one_hot(e[in_distribution])
    X_ood = X[~in_distribution]
    y_ood = y[~in_distribution]

    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X_train_test, y_train_test, e_train_test, test_size=test_size, random_state=seed
    )
    e_ood = np.zeros((len(X_ood), e_train_test.shape[1]), dtype="float64")

    return X_train, X_test, X_ood, y_train, y_test, y_ood, e_train, e_test, e_ood


y = df["labels"]
for i, u in enumerate(y.unique()):
    y[y == u] = i

y = y.astype(int).to_numpy()
e = df["region"].astype("category").cat.codes.to_numpy()
X = df.drop(["labels", "region"], axis=1).to_numpy()

regions = [
    "Ariege",
    "Aude",
    "Aveyron",
    "Gard",
    "Gers",
    "HauteGaronne",
    "HautesPyrenees",
    "Herault",
    "Lot",
    "Lozere",
    "PyreneesOrientales",
    "Tarn",
    "TarnEtGaronne",
]

e_id = 4
print(f"\n\nEnvironment {e_id}\n")

(
    X_train,
    X_test,
    X_ood,
    y_train,
    y_test,
    y_ood,
    e_train,
    e_test,
    e_ood,
) = train_test_ood_split(X, y, e, e_id)

anchor_loss = AnchorClassificationLoss(gamma=5, n_classes=len(np.unique(y)))
likelihood = AnchorClassificationLoss(gamma=1, n_classes=len(np.unique(y)))

train_data = lgb.Dataset(
    X_train,
    y_train,
    # init_score=anchor_loss.init_score(y_train),
)
train_data.anchor = e_train

test_data = lgb.Dataset(
    X_test,
    y_test,
    init_score=anchor_loss.init_score(y_test),
)
test_data.anchor = e_test

ood_data = lgb.Dataset(
    X_ood,
    y_ood,
    init_score=anchor_loss.init_score(y_ood),
)
ood_data.anchor = e_ood


def classification_score(f, data):
    y = data.get_label()
    f = np.reshape(f, (len(y), -1), order="F")
    predictions = np.argmax(f, axis=1)
    return "accuracy", np.mean(predictions == y), True


lgbm_model = lgb.train(
    params={
        "learning_rate": 0.2,
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
    },
    train_set=train_data,
    num_boost_round=100,
    valid_sets=(train_data, test_data, ood_data),
    valid_names=("train", "test", "ood"),
    verbose_eval=20,
    feval=(classification_score, likelihood.score),
)

control_model = lgb.train(
    params={"learning_rate": 0.2, "num_class": len(np.unique(y))},
    train_set=train_data,
    num_boost_round=100,
    valid_sets=(train_data, test_data, ood_data),
    valid_names=("train", "test", "ood"),
    verbose_eval=20,
    fobj=anchor_loss.objective,
    feval=(classification_score, likelihood.score),
)

anchor_model = lgb.train(
    params={"learning_rate": 0.2, "num_class": len(np.unique(y))},
    train_set=train_data,
    num_boost_round=100,
    valid_sets=(train_data, test_data, ood_data),
    valid_names=("train", "test", "ood"),
    verbose_eval=20,
    fobj=anchor_loss.objective,
    feval=(classification_score, likelihood.score),
)

breakpoint()
lgbm_residuals = anchor_loss.residuals(lgbm_model.predict(X), y)
anchor_residuals = anchor_loss.residuals(
    anchor_loss.predictions(anchor_model.predict(X)), y
)
data = pd.DataFrame(
    {
        "e": e,
        **{f"lgbm_residuals_{i}": lgbm_residuals[:, i] for i in range(6)},
        **{f"anchor_residuals_{i}": anchor_residuals[:, i] for i in range(6)},
        "y": y,
    }
)
data.groupby("e").mean()


fig, axes = plt.subplots(2, 3, figsize=(10, 5))

colors = ["b", "r", "g", "c", "m", "y"]
bins = np.linspace(-1, 1, 20)
for i in range(3):
    lgbm_residuals = anchor_loss.residuals(lgbm_model.predict(X[e == i]), y[e == i])
    anchor_residuals = anchor_loss.residuals(
        anchor_loss.predictions(anchor_model.predict(X[e == i])), y[e == i]
    )
    for outcome in range(6):
        axes[0, i].hist(
            lgbm_residuals[:, outcome], color=colors[outcome], bins=bins, alpha=0.3
        )
        axes[1, i].hist(
            anchor_residuals[:, outcome], color=colors[outcome], bins=bins, alpha=0.3
        )
        axes[0, i].set_title(f"Environment {i}")

    axes[0, i].vlines(lgbm_residuals.mean(axis=0), 0, 1000, color=colors)
    axes[1, i].vlines(anchor_residuals.mean(axis=0), 0, 1000, color=colors)

plt.show()
