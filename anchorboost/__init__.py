from .anchor_mixins import AnchorMixin
from .anchor_objectives import (
    AnchorKookClassificationObjective,
    AnchorKookMultiClassificationObjective,
    AnchorRegressionObjective,
)
from .classification_mixins import ClassificationMixin, MultiClassificationMixin
from .lgbm_mixins import LGBMMixin
from .regression_mixins import RegressionMixin

# backwards compatability
AnchorClassificationLoss = AnchorKookClassificationObjective
AnchorRegressionLoss = AnchorRegressionObjective

__all__ = [
    "AnchorKookClassificationObjective",
    "AnchorKookMultiClassificationObjective",
    "AnchorClassificationLoss",
    "AnchorRegressionLoss",
    "AnchorMixin",
    "AnchorRegressionObjective",
    "ClassificationMixin",
    "LGBMMixin",
    "MultiClassificationMixin",
    "RegressionMixin",
]
