"""
    IAML (Incremental AutoML), a high-performance and modular,
    open-source Python framework. Designed to mimics the behavior
    of a data scientist in creating pipelines and leverages an 
    optimization process inspired by genetic algorithm for
    efficient pipeline construction and hyperparameter tuning.
    
    The framework incorporates explainability features, such as 
    SHAP-based insights, to enhance model transparency and trustworthiness.
"""
from .iaml import IAML
from .core_dispatcher import CoreDispatcher
from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .candidate import Candidate
from .dataset import Dataset
from .data_type import DataType
from .metric_plot import MetricPlot
from .metric import Metric
from .plot import Plot, StatisticPlot
from .statistic import Statistic
from .cache import Cache
from .meta_predictor import MetaPredictor
from .void_step import VoidStep

from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep
from .meta_partial_explorer_step import MetaPartialExplorerStep

# Default Actionables
from .actionables import *

# Wrappers
from .wrapper import *

# Metrics
from .metrics import *

# Statistics
from .statistics import *

# Plots
from .plots import *

# Stack
from .stack import Stack

# Optimizer
from .optimizers import *

# Meta predictors
from .meta_predictors import *

# Type of target
from .type_of_target import type_of_target

# Ensure star imports expose all public names, even if __all__ is set elsewhere.
__all__ = [
    name for name in globals()
    if not name.startswith("_") and name != "__all__"
]
