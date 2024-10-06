"""
[PLOT] Confusion Matrix Plot
"""
import textwrap
import io
import pandas as pd
from yellowbrick.classifier import ConfusionMatrix

from ..plot import Plot, capture

class ConfusionMatrixPlot(Plot):
    """
    [PLOT] Confusion Matrix Plot
    """
    
    title = "Confusion Matrix"
    description = textwrap.dedent("""
        The Confusion Matrix is a powerful visualization tool used to assess how well a classification model 
        is performing, particularly in identifying different medical conditions or diagnostic categories. 
        It provides a clear breakdown of true positive, false positive, true negative, and false negative rates.

        For example, in a healthcare setting, if a model is trained to classify whether a patient has 'diabetes', 
        'hypertension', or is 'healthy', the Confusion Matrix will show how many times the model made correct predictions 
        (true positives and true negatives) and where it made mistakes (false positives and false negatives).

        Each row in the matrix represents the actual condition of the patient, and each column represents the predicted 
        condition. This makes it easy to spot patterns in the model's predictions, such as whether it tends to misclassify 
        one condition as another.

        Doctors and data scientists rely on this plot to understand not only how often the model is right but also the 
        types of errors it makes. This information is crucial in healthcare, where reducing misdiagnoses can significantly 
        improve patient outcomes.
        """)
    
    @capture
    def compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y:pd.DataFrame, **kwargs) -> Plot:
        """
        Compute plot given X, y.
        """
        self._binary_image = io.BytesIO()
        self.__visualizer = ConfusionMatrix(estimator, is_fitted=True)
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
