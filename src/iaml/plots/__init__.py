"""
All plots
"""

# Classifier
from .class_prediction_error_plot import ClassPredictionErrorPlot
from .classification_report_plot import ClassificationReportPlot
from .confusion_matrix_plot import ConfusionMatrixPlot
from .rocauc_plot import ROCAUCPlot
from .precision_recall_curve_plot import PrecisionRecallCurvePlot

# Regressor 
from .residual_plot import ResidualsPlot
from .prediction_error_plot import PredictionErrorPlot

# Survival
from .kaplan_meier_comparison_plot import KaplanMeierModelComparisonPlot
from .cumulative_hazard_plot import CumulativeHazardModelComparisonPlot
from .roc_dynamique_curve_plot import ROCDynamiqueCurvePlot
from .shap_plot import ShapPlot
