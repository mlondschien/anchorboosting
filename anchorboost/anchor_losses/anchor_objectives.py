import numpy as np

from anchorboost.anchor_losses import (
    AnchorMixin,
    ClassificationMixin,
    MultiClassificationMixin,
)


class AnchorKookMultiClassificationObjective(AnchorMixin, MultiClassificationMixin):
    def __init__(self, gamma, n_classes):
        self.gamma = gamma
        super().__init__(n_classes)

    def residuals(self, f, data):
        predictions = self.predictions(f)
        predictions[self._indices(data.get_label())] -= 1
        return predictions.flatten("F")

    def loss(self, f, data):
        residuals = self.residuals(f, data).reshape((-1, self.n_classes), order="F")
        proj_residuals = self._proj(data.anchor, residuals)
        return super().loss(f, data) + (self.gamma - 1) * np.sum(
            proj_residuals**2, axis=1
        )

    def grad(self, f, data):
        residuals = self.residuals(f, data).reshape((-1, self.n_classes), order="F")
        proj_residuals = self._proj(data.anchor, residuals)
        predictions = self.predictions(f)
        proj_residuals -= np.sum(proj_residuals * predictions, axis=1, keepdims=True)
        anchor_grad = 2 * (self.gamma - 1) * predictions * proj_residuals
        return super().grad(f, data) + anchor_grad.flatten("F")


class AnchorKookClassificationObjective(AnchorMixin, ClassificationMixin):
    def __init__(self, gamma):
        self.gamma = gamma

    def residuals(self, f, data):
        return self.predictions(f) - data.get_label()

    def loss(self, f, data):
        return (
            super().loss(f, data)
            + (self.gamma - 1) * self._proj(data.anchor, self.residuals(f, data)) ** 2
        )

    def grad(self, f, data):
        return super().grad(f, data) + 2 * (self.gamma - 1) * self._proj(
            data.anchor, self.residuals(f, data)
        ) * self.residuals(f, data)
