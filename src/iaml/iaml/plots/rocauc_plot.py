"""[PLOT] ROC-AUC Plot"""
from __future__ import annotations

import textwrap
import io
from typing import TYPE_CHECKING
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ..metric_plot import MetricPlot, capture
if TYPE_CHECKING:
    from ..iaml_pipeline import IAMLPipeline


class ROCAUCPlot(MetricPlot):
    """[PLOT] ROC-AUC Plot"""

    title: str = "Receiver Operating Characteristic - Area Under the Curve"
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

        **Micro-average** and **macro-average** ROC curves are useful when dealing with multiclass classification 
        problems (where there are more than two classes). 

        - **Micro-average** ROC aggregates the contributions of all classes and calculates metrics globally 
            by counting the total true positives, false positives, true negatives, and false negatives. 
            It provides a single ROC curve by combining all classes, treating each decision as a binary one 
            (one-vs-rest). This is useful when you care about the overall performance of the classifier across 
            all categories.

        - **Macro-average** ROC computes the ROC curve for each class separately and then averages the results. 
            This gives equal weight to all classes, regardless of the number of samples. Macro-average is useful 
            when you want to evaluate the model's performance on each class individually, giving each class the 
            same importance, regardless of how often it appears in the dataset.

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

        pos_label = None
        if not (y.dtype in ['int', 'bool']):
            pos_label = y.iloc[0] if isinstance(y, pd.Series) else y[0]

        # Predict probabilities
        y_prob = estimator.predict_proba(X)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_prob, pos_label=pos_label)
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
