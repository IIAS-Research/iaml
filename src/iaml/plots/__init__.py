"""All plots"""

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

# Descriptive statistics
from .bar_plot import BarPlot
from .line_plot import LinePlot
from .histogram_plot import HistogramPlot
from .box_plot import BoxPlot
from .violin_plot import ViolinPlot
from .density_plot import DensityPlot
from .qq_plot import QQPlot
from .correlation_heatmap_plot import CorrelationHeatmapPlot
from .missingness_heatmap_plot import MissingnessHeatmapPlot
from .pair_plot import PairPlot
from .target_distribution_plot import TargetDistributionPlot
from .outlier_plot import OutlierPlot
