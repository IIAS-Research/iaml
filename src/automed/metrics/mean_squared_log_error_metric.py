"""
[METRIC] Mean Squared Log Error
"""
from sklearn.metrics import mean_squared_log_error
import pandas as pd
from ..metric import Metric


class MeanSquaredLogErrorMetric(Metric):
    """
    [METRIC] Mean Squared Log Error
    """
    def __str__(self):
        return 'mean_squared_log_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Mean squared logarithmic error regression loss.'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this input ?
        Must be regression with no negative value

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target == 'continuous' and not (y < 0).any(axis=None)
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        try:
            return mean_squared_log_error(y, y_pred)
        except:  # pylint: disable=bare-except
            return None
