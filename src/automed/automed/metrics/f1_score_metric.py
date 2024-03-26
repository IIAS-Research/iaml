"""
[METRIC] F1 Score
"""
import pandas as pd
from sklearn.utils.multiclass import type_of_target as sk_type_of_target
from sklearn.metrics import f1_score
from ..metric import Metric

class F1ScoreMetric(Metric):
    """
    [METRIC] F1 Score
    """
    def __str__(self):
        return 'f1_score'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Compute the F1 score, also known as balanced F-score or F-measure. \
            The F1 score can be interpreted as a harmonic mean of the precision and recall'
    
            
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
            # TODO Find something less arbitrary (about pos_label)
            return f1_score(y, y_pred, pos_label=y[0])
        if sk_type_of_target(y) == 'multiclass':
            return f1_score(y, y_pred, average ='weighted')
        if sk_type_of_target(y) == 'multilabel-indicator':
            return f1_score(y, y_pred, average ='samples')
        
        raise ValueError('Metric not suitable') 
