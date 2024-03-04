"""
[METRIC] R2 Score
"""
import pandas as pd
from sklearn.metrics import r2_score
from ..metric import Metric

class R2ScoreMetric(Metric):
    """
    [METRIC] R2 Score
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'R2(coefficient of determination) regression score function. \
            Best possible score is 1.0 and it can be negative \
            (because the model can be arbitrarily worse).'
    
    def __str__(self):
        return 'r2_score'
        
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
        return r2_score(y, y_pred)
