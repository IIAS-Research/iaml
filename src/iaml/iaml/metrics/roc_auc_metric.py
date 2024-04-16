"""
[METRIC] ROC AUC
"""
from sklearn.metrics import roc_auc_score
import pandas as pd
from ..metric import Metric

class RocAucMetric(Metric):
    """
    [METRIC] ROC AUC
    """
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Compute the ROC AUC.'
    
    def __str__(self):
        return 'ROC AUC'
    
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
        return type_of_target == 'binary'
    
    def need_proba(self):
        return True
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return roc_auc_score(y, y_pred[:, 1])
