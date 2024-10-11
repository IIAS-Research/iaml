"""
[PLOT] Precision-Recall Curve Plot
"""
import textwrap
import io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from ..plot import MetricPlot, capture

class PrecisionRecallCurvePlot(MetricPlot):
    """
    [PLOT] Precision-Recall Curve Plot
    """
    
    title = "Precision-Recall Curve"
    description = textwrap.dedent("""
        The Precision-Recall Curve is a valuable visualization tool for evaluating the performance of 
        a classification model, particularly in healthcare where identifying the correct balance between 
        precision (positive predictive value) and recall (sensitivity or true positive rate) is critical.

        Precision measures how many of the predicted positive cases (e.g., disease diagnoses) were actually correct, 
        while recall indicates how well the model identifies all the true positive cases. The Precision-Recall Curve 
        shows this trade-off across different threshold settings of the model.

        In healthcare, for example, if a model is predicting whether a patient has a disease, the Precision-Recall Curve 
        will help you understand how the model performs when prioritizing minimizing false positives (increasing precision) 
        versus maximizing the detection of true positives (increasing recall). This is especially important in imbalanced 
        datasets where one condition (e.g., healthy patients) dominates over others (e.g., rare diseases).

        Doctors and data scientists rely on this visualization to optimize the model based on specific healthcare priorities, 
        such as minimizing missed diagnoses or reducing unnecessary treatments, making the Precision-Recall Curve an essential 
        tool for improving patient outcomes.
        """)
    
    @capture
    def _compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y,
            X_train:pd.DataFrame=None, y_train=None,
            **kwargs) -> MetricPlot:
        """
        Compute plot given X, y.
        """
        self._binary_image = io.BytesIO()
        
        pos_label = y.iloc[0] if isinstance(y, pd.Series) else y[0]
        
        # Predict probabilities for the positive class
        y_prob = estimator.predict_proba(X)[:, 1]
        
        # Compute Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_prob, pos_label=pos_label)
        average_precision = average_precision_score(y, y_prob, pos_label=pos_label)
        
        # Create the Precision-Recall plot
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        # Save plot to binary image
        plt.savefig(self._binary_image, format='png')
        plt.close()
        
        return self
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target?
        """
        return type_of_target in ['binary']
