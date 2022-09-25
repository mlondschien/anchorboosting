import numpy as np

from anchorboost.anchor_mixins import AnchorMixin
from anchorboost.classification_mixins import (
    ClassificationMixin,
    MultiClassificationMixin,
)
from anchorboost.lgbm_mixins import LGBMMixin
from anchorboost.regression_mixins import RegressionMixin


class AnchorKookClassificationObjective(AnchorMixin, ClassificationMixin, LGBMMixin):
    def __init__(self, gamma, center_residuals=False):
        self.gamma = gamma
        self.center_residuals = center_residuals

    def residuals(self, f, data):
        residuals = self.predictions(f) - data.get_label()
        if self.center_residuals:
            residuals -= residuals.mean()
        return residuals

    def loss(self, f, data):
        return (
            super().loss(f, data)
            + (self.gamma - 1) * self._proj(data.anchor, self.residuals(f, data)) ** 2
        )

    def grad(self, f, data):
        predictions = self.predictions(f)
        proj_residuals = self._proj(data.anchor, self.residuals(f, data))

        if self.center_residuals:
            proj_residuals -= proj_residuals.mean()

        return super().grad(f, data) + 2 * (
            self.gamma - 1
        ) * proj_residuals * predictions * (1 - predictions)


class AnchorKookMultiClassificationObjective(
    AnchorMixin, MultiClassificationMixin, LGBMMixin
):
    def __init__(self, gamma, n_classes, center_residuals=False):
        self.gamma = gamma
        self.center_residuals = center_residuals
        super().__init__(n_classes)

    def residuals(self, f, data):
        predictions = self.predictions(f)
        predictions[self._indices(data.get_label())] -= 1
        if self.center_residuals:
            predictions -= predictions.mean(axis=0, keepdims=True)

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

        if self.center_residuals:
            proj_residuals -= proj_residuals.mean(axis=0, keepdims=True)

        predictions = self.predictions(f)
        proj_residuals -= np.sum(proj_residuals * predictions, axis=1, keepdims=True)
        anchor_grad = 2 * (self.gamma - 1) * predictions * proj_residuals
        return super().grad(f, data) + anchor_grad.flatten("F")


class AnchorRegressionObjective(AnchorMixin, RegressionMixin, LGBMMixin):
    def __init__(self, gamma):
        self.gamma = gamma

    def residuals(self, f, data):
        return data.get_label() - f

    def loss(self, f, data):
        return (
            super().loss(f, data)
            + (self.gamma - 1) * self._proj(data.anchor, self.residuals(f, data)) ** 2
        )

    def grad(self, f, data):
        return super().grad(f, data) + 2 * (self.gamma - 1) * self._proj(
            data.anchor, self.residuals(f, data)
        ) * (-1)
