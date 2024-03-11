"""
[STEP] Learn :  Linear Regression
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('learning', 'tabular', 'fast_learning')
class ActLinearRegression(Predictor):
    """
    [STEP] Learn :  Linear Regression
    """
    name = "Learn : Linear Regression"
    def __init__(self):
        self.configuration:dict = {}
        self.model:LinearRegression = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Linear regression on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = LinearRegression()
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
    
    def suitable(self, candidate: Candidate) -> bool:
        return candidate.dataset.type_of_target in ['continuous']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
