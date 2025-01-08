from .anchor_objectives import (
    AnchorHSICRegressionObjective,
    AnchorKookClassificationObjective,
    AnchorKookMultiClassificationObjective,
    AnchorLiuClassificationObjective,
    AnchorRegressionObjective,
)
from .classification_mixins import ClassificationMixin, MultiClassificationMixin
from .lgbm_mixins import LGBMMixin
from .regression_mixins import RegressionMixin

# backwards compatability
AnchorClassificationLoss = AnchorKookClassificationObjective
AnchorRegressionLoss = AnchorRegressionObjective

__all__ = [
    "AnchorClassificationLoss",
    "AnchorHSICRegressionObjective",
    "AnchorKookClassificationObjective",
    "AnchorKookMultiClassificationObjective",
    "AnchorLiuClassificationObjective",
    "AnchorRegressionLoss",
    "AnchorRegressionObjective",
    "ClassificationMixin",
    "LGBMMixin",
    "MultiClassificationMixin",
    "RegressionMixin",
]
