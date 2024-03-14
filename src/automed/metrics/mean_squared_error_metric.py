"""
[METRIC] Mean Squared Error
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from ..metric import Metric

class MeanSquaredErrorMetric(Metric):
    """
    [METRIC] Mean Squared Error
    """
    def __str__(self):
        return 'mean_squared_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Mean squared error regression loss.'
        
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be regression

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target == 'continuous'
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return mean_squared_error(y, y_pred)
