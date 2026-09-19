"""Integrated AutoML for Medical Labs (IAML).

IAML helps clinical research teams build, evaluate and inspect machine learning
pipelines for tabular classification, regression and survival analysis. Modular
preprocessing and modeling steps support automated search, while pipeline
descriptions, evaluation metrics and SHAP explanations support review of the
resulting models and reporting of research methods.
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

# Type of target
from .type_of_target import type_of_target

# Ensure star imports expose all public names, even if __all__ is set elsewhere.
__all__ = [
    name for name in globals()
    if not name.startswith("_") and name != "__all__"
]
