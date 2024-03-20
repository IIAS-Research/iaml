"""
[METRIC] Classification Error
"""
import pandas as pd
from .balanced_accuracy_metric import BalancedAccuracyMetric
from ..metric import Metric

class ClassificationErrorMetric(Metric):
    """
    [METRIC] Classification Error
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the classification error, if accuracy is relevant then the \
            classification error is calculated by 1 - accuracy otherwise if balanced \
            accuracy is relevant then the classification error is \
            calculated by 1 - balanced _accuracy.'
    
    def __str__(self):
        return 'classification_error'
    
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
        return type_of_target in ['binary', 'multiclass', 'multilabel-indicator']
    
    # Calculate the classification error using either accuracy or balanced accuracy, 
    # depending on relevence
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        balanced_accuracy = BalancedAccuracyMetric().compute(y, y_pred)
        return 1 - balanced_accuracy
