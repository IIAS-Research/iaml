"""
[PLOT] Class Prediction Error Plot
"""
import textwrap
import io
import pandas as pd
from yellowbrick.classifier import ClassPredictionError

from ..plot import Plot, capture

class ClassPredictionErrorPlot(Plot):
    """
    [PLOT] Class Prediction Error Plot
    """
    
    @property
    def explain(self) -> str:
        """
        Return str explanation of the plot
        """
        return textwrap.dedent("""
            The Class Prediction Error is a visualization that helps understand how well a machine 
            learning model is performing in predicting medical conditions or diagnoses. It shows 
            both the correct predictions made by the model and the mistakes it makes for each condition.

            For example, imagine you have a model that’s trained to identify different diseases 
            from patient data, such as predicting whether someone has diabetes, hypertension, 
            or is healthy. The Class Prediction Error plot would show, for each of these conditions, 
            how many times the model correctly identified the disease and how many times it made a 
            wrong prediction.

            For instance, if the model predicts "diabetes" for a patient who actually has "hypertension," 
            the plot will highlight this error. Similarly, it will also show how often the model 
            correctly identifies "healthy" patients versus when it mistakenly predicts they have 
            a disease.

            This visualization is especially helpful for doctors and data scientists because it clearly 
            shows where the model is making errors, making it easier to improve its accuracy, which is 
            critical in healthcare where correct predictions can have a big impact on patient outcomes.
            """)
    
    @capture
    def compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y:pd.DataFrame, **kwargs) -> Plot:
        """
        Compute plot given X, y. 
        """
        self._binary_image = io.BytesIO()
        self.__visualizer = ClassPredictionError(estimator, is_fitted=True)
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
