"""
[METRIC] Specificity
"""
import pandas as pd
from sklearn.metrics import confusion_matrix
from ..metric import Metric

class SpecificityMetric(Metric):
    """
    [METRIC] Specificity
    """
    def __str__(self):
        return 'specificity'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'The proportion of negative instances that are corectely \
            classified as negative: tn / (tn + fp)'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be classification

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['binary']

    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0
