"""
[PLOT] ROC-AUC Plot
"""
import textwrap
import io
import pandas as pd
from yellowbrick.classifier import ROCAUC

from ..plot import Plot, capture

class ROCAUCPlot(Plot):
    """
    [PLOT] ROC-AUC Plot
    """
    
    @property
    def explain(self) -> str:
        """
        Return str explanation of the plot
        """
        return textwrap.dedent("""
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
    def compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y:pd.DataFrame, **kwargs) -> Plot:
        """
        Compute plot given X, y.
        """
        self._binary_image = io.BytesIO()
        self.__visualizer = ROCAUC(estimator, is_fitted=True)
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return type_of_target in ['binary', 'multiclass', 'multilabel-indicator']
