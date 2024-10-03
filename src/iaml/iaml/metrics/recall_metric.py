"""
[METRIC] Recall
"""
import pandas as pd
from sklearn.metrics import recall_score
from ..metric import Metric
from ..type_of_target import type_of_target  as get_type_of_target

class RecallMetric(Metric):
    """
    [METRIC] Recall
    """
    
    def __str__(self):
        return 'recall'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'The recall is the ratio tp / (tp + fn) where tp is the number \
            of true positives and fn the number of false negatives. The recall is \
            intuitively the ability of the classifier to find all the positive samples.'
    
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
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        if get_type_of_target(y) == 'binary':
            # TODO Find something less arbitrary
            return recall_score(y, y_pred, pos_label=y[0], zero_division=0.0)
        if get_type_of_target(y) == 'multiclass':
            return recall_score(y, y_pred, average = 'weighted', zero_division=0.0) 
        if get_type_of_target(y) == 'multilabel-indicator':
            return recall_score(y, y_pred, average= 'samples', zero_division=0.0)
        
        raise ValueError('Metric not suitable')
