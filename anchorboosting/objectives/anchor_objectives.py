import numpy as np

from anchorboosting.objectives.mixins import (
    ClassificationMixin,
    LGBMMixin,
    MultiClassificationMixin,
    ProjMixin,
    RegressionMixin,
)


class AnchorKookClassificationObjective(ClassificationMixin, LGBMMixin, ProjMixin):
    def __init__(self, gamma, precompute_proj=True, n_categories=None):
        super().__init__(precompute_proj=precompute_proj, n_categories=n_categories)
        self.gamma = gamma
        self.name = "kook anchor classification"

    def __repr__(self):
        return f"AnchorKookClassificationObjective(gamma={self.gamma})"

    def residuals(self, f, data):
        residuals = self.predictions(f) - data.get_label()
        return residuals

    def loss(self, f, data):
        return (
            super().loss(f, data)
            + (self.gamma - 1)
            * self.proj(
                data.anchor,
                self.residuals(f, data),
                copy=False,
            )
            ** 2
        )

    def grad(self, f, data):
        predictions = self.predictions(f)
        proj_residuals = self.proj(data.anchor, self.residuals(f, data), copy=False)

        return super().grad(f, data) + 2 * (
            self.gamma - 1
        ) * proj_residuals * predictions * (1 - predictions)


class AnchorKookMultiClassificationObjective(
    MultiClassificationMixin, LGBMMixin, ProjMixin
):
    def __init__(self, gamma, n_classes, precompute_proj=True, n_categories=None):
        self.gamma = gamma
        super().__init__(
            n_classes=n_classes,
            precompute_proj=precompute_proj,
            n_categories=n_categories,
        )
        self.name = "kook anchor multi-classification"

    def __repr__(self):
        return f"AnchorKookMultiClassificationObjective(gamma={self.gamma})"

    def residuals(self, f, data):
        predictions = self.predictions(f)
        predictions[self._indices(data.get_label())] -= 1
        return predictions.flatten("F")

    def loss(self, f, data):
        residuals = self.residuals(f, data).reshape((-1, self.n_classes), order="F")
        proj_residuals = self.proj(data.anchor, residuals, copy=False)
        # Multiply with self.factor to align two-class classification with
        # AnchorKookClassificationObjective
        return super().loss(f, data) + self.factor * (self.gamma - 1) * np.sum(
            proj_residuals**2, axis=1
        )

    def grad(self, f, data):
        residuals = self.residuals(f, data).reshape((-1, self.n_classes), order="F")
        proj_residuals = self.proj(data.anchor, residuals, copy=False)

        predictions = self.predictions(f)
        proj_residuals -= np.sum(proj_residuals * predictions, axis=1, keepdims=True)
        # Multiply with factor to align two-class classification with
        # AnchorKookClassificationObjective
        anchor_grad = 2 * self.factor * (self.gamma - 1) * predictions * proj_residuals
        return super().grad(f, data) + anchor_grad.flatten("F")


class AnchorRegressionObjective(RegressionMixin, LGBMMixin, ProjMixin):
    def __init__(self, gamma, n_categories=None, precompute_proj=True):
        super().__init__(precompute_proj=precompute_proj, n_categories=n_categories)
        self.gamma = gamma
        self.name = "anchor regression"

    def __repr__(self):
        return f"AnchorRegressionObjective(gamma={self.gamma})"

    def residuals(self, f, data):
        return data.get_label() - f

    def loss(self, f, data):
        if self.gamma == 1:
            return super().loss(f, data)

        # For gamma <= 1, this is equivalent to kappa := (gamma - 1) / gamma and
        # loss = (1 - kappa) | y - f |^2 + kappa | P_Z (y - f) |^2
        return (
            super().loss(f, data)
            + 0.5
            * (self.gamma - 1)
            * self.proj(
                data.anchor,
                self.residuals(f, data),
                copy=False,
            )
            ** 2
        )

    def grad(self, f, data):
        residuals = self.residuals(f, data)
        if self.gamma == 1:
            return -residuals

        proj_residuals = self.proj(data.anchor, residuals, copy=True)
        return -residuals - (self.gamma - 1) * proj_residuals
