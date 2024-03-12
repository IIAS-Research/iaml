"""
[METRIC] Precision
"""
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.utils.multiclass import type_of_target as sk_type_of_target
from ..metric import Metric

class PrecisionMetric(Metric):
    """
    [METRIC] Precision
    """
    def __str__(self):
        return 'precision'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Compute the precision: The precision is the ratio tp / (tp + fp) \
            where tp is the number of true positives and fp the number of false positives.'
    
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
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        if sk_type_of_target(y) == 'binary':
            # TODO Find something less arbitrary
            return precision_score(y, y_pred, pos_label=y[0], zero_division=0.0)
        if sk_type_of_target(y) == 'multiclass':
            return precision_score(y, y_pred, average = 'weighted', zero_division=0.0) 
        if sk_type_of_target(y) == 'multilabel-indicator':
            return precision_score(y, y_pred, average= 'samples', zero_division=0.0)
        
        raise ValueError('Metric not suitable') 
