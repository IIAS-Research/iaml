"""[PLOT] Classification Report Plot"""
import textwrap
import io
from typing import TYPE_CHECKING
import pandas as pd
from yellowbrick.classifier import ClassificationReport

from ..plot import MetricPlot, capture
if TYPE_CHECKING:
    from ..iaml_pipeline import IAMLPipeline


class ClassificationReportPlot(MetricPlot):
    """[PLOT] Classification Report Plot"""
    
    title: str = "Classification report"
    description: str = textwrap.dedent("""
        The Classification Report is a visual tool to evaluate the performance of a machine learning model 
        on classification tasks, such as diagnosing medical conditions. This plot provides key metrics for 
        each class (like diseases or health conditions) that the model is trained to identify.

        It includes metrics such as precision, recall, F1-score, and support for each class. These metrics 
        are essential to understanding how well the model is identifying true positives (correct diagnoses), 
        minimizing false positives (incorrect diagnoses), and balancing between precision and recall.

        For example, if you have a model classifying conditions like 'healthy', 'diabetes', and 'hypertension', 
        the Classification Report plot will show you the precision (how many of the predicted conditions were correct), 
        recall (how many actual conditions were correctly identified), and the F1-score (the harmonic mean of precision 
        and recall). This is especially important in healthcare to ensure the model provides balanced and accurate results 
        across all classes, improving both diagnosis and patient outcomes.

        Doctors and data scientists use this visualization to easily compare the model's performance on different conditions, 
        aiding in model refinement and ensuring robust diagnostic predictions.
        """)
    
    @capture
    def _compute(
        self,
        estimator: 'IAMLPipeline', 
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_train: pd.DataFrame = None,
        y_train: pd.DataFrame = None,
        **kwargs) -> MetricPlot:
        self._binary_image = io.BytesIO()
        self.__visualizer = ClassificationReport(estimator, is_fitted=True)
        
        if X_train is not None and y_train is not None:
            self.__visualizer.fit(X_train, y_train)
            
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
