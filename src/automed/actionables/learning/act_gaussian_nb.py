"""
[STEP] Learn : Gaussian NB
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...output import Input
from ...decorators.all import is_step

@is_step('learning', 'tabular')
class ActGaussianNb(Actionable):
    """
    [STEP] Learn : Gaussian NB
    """
    name = "Learn : Gaussian NB"
    def __init__(self):
        self.configuration:dict = {}
        self.model:GaussianNB = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit GaussianNB on Input.dataset

        Args:
            dataset (Input): Fit data

        Returns:
            Fitted step
        """
        self.model = GaussianNB()
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
    
    def suitable(self, input_data:Input) -> bool:
        """
        Does this step suitable for this input

        Args:
            input_data (Input): Suitable for this input

        Returns:
            bool: Suitable ?
        """
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
