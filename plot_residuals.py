import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from anchorboosting import (
    AnchorKookClassificationObjective,
    AnchorLiuClassificationObjective,
)

liu = AnchorLiuClassificationObjective(1)
kook = AnchorKookClassificationObjective(1)

f = np.linspace(-20, 20, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(f, liu.residuals(f, lgb.Dataset(None, 0)), color="red", label="y=0")
axes[0].plot(f, liu.residuals(f, lgb.Dataset(None, 1)), color="blue", label="y=1")
axes[0].plot(f, -f, "k--", label="-id")
axes[0].set_title("Liu")
axes[0].legend()

axes[1].plot(f, -kook.residuals(f, lgb.Dataset(None, 0)), color="red", label="y=0")
axes[1].plot(f, -kook.residuals(f, lgb.Dataset(None, 1)), color="blue", label="y=1")
axes[1].set_title("Kook")
axes[1].legend()
plt.show()
