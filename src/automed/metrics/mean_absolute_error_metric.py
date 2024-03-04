"""
[METRIC] Mean Absolute Error
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from ..metric import Metric

class MeanAbsoluteErrorMetric(Metric):
    """
    [METRIC] Mean Absolute Error
    """
    def __str__(self):
        return 'mean_absolute_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Mean absolute error regression loss.'
        
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this input ?
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
        return mean_absolute_error(y, y_pred)