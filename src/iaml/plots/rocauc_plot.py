"""[PLOT] ROC-AUC Plot"""
from __future__ import annotations

import textwrap
import io
from typing import TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ..metric_plot import MetricPlot, capture
from ._classification import binary_score_inputs
if TYPE_CHECKING:
    from ..iaml_pipeline import IAMLPipeline


class ROCAUCPlot(MetricPlot):
    """[PLOT] ROC-AUC Plot"""

    title: str = "Receiver Operating Characteristic - Area Under the Curve"
    needed_prediction = "decision_function"
    description: str = textwrap.dedent("""
        The ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) plot is a widely used 
        tool to assess the performance of a classification model, especially in the healthcare domain. 
        It provides a graphical representation of the model's ability to distinguish between classes, 
        such as diagnosing the presence or absence of a medical condition.

        The ROC curve shows the trade-off between the true positive rate (sensitivity) and false positive 
        rate for different threshold values. The AUC score (Area Under the Curve) summarizes the performance 
        into a single number, where a value of 1 indicates perfect classification, and 0.5 represents a 
        model no better than random guessing.

        For instance, in a medical setting, you might have a model predicting whether a patient has a 
        certain disease or is healthy. The ROC curve would help evaluate how well the model can separate 
        patients with the disease from those without. The closer the curve is to the top-left corner and 
        the higher the AUC score, the better the model is at distinguishing between the conditions.

        This plot supports binary classifiers. The positive class is the second fitted class,
        scored by decision_function or the second predict_proba column. Both classes must occur in the
        evaluation targets for ROC AUC to be defined.

        Doctors and data scientists use this visualization to ensure that the model performs well across 
        different threshold values, making it a critical tool in situations where misdiagnosis could have 
        serious consequences.
        """)

    @capture
    def compute(
        self,
        estimator: IAMLPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        **kwargs) -> MetricPlot:
        self._binary_image = io.BytesIO()

        targets, y_prob = binary_score_inputs(estimator, X, y)
        if not targets.any() or targets.all():
            raise ValueError("ROC AUC requires both fitted classes in the evaluation targets")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets, y_prob)
        roc_auc = auc(fpr, tpr)

        # Create the ROC plot
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Save plot to binary image
        plt.savefig(self._binary_image, format='png')

        return self

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:
        return type_of_target == 'binary'
