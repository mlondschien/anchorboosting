import numpy as np

from anchorboost.anchor_losses import (
    AnchorMixin,
    ClassificationMixin,
    MultiClassificationMixin,
)


class AnchorKookMultiClassificationLoss(AnchorMixin, MultiClassificationMixin):
    def residuals(self, f, data):
        predictions = self.predictions(f)
        predictions[self._indices(data.get_label())] -= 1
        return predictions.flatten("F")

    def loss(self, f, data):
        return (
            super(MultiClassificationMixin).loss(f, data)
            + (self.gamma - 1) * self._proj(data.anchor, self.residuals(f, data)) ** 2
        )

    def grad(self, f, data):
        return super(MultiClassificationMixin).grad(f, data) + 2 * (
            self.gamma - 1
        ) * np.sum(
            self._proj(data.anchor, self.residuals(f, data)) * self.residuals(f, data),
            axis=1,
        )


class AnchorKookClassificationLoss(AnchorMixin, ClassificationMixin):
    def residuals(self, f, data):
        return self.predictions(f) - data.get_label()

    def loss(self, f, data):
        return (
            super(ClassificationMixin).loss(f, data)
            + (self.gamma - 1) * self._proj(data.anchor, self.residuals(f, data)) ** 2
        )

    def grad(self, f, data):
        return super(ClassificationMixin).grad(f, data) + (self.gamma - 1) * self._proj(
            data.anchor, self.residuals(f, data)
        ) * self.residuals(f, data)
