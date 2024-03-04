"""
[METRIC] Accuracy
"""
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd
from ..metric import Metric

class AccuracyMetric(Metric):
    """
    [METRIC] Accuracy
    """
    def __str__(self) -> str:
        return 'accuracy'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Accuracy classification score.'
    
    # Get one label (numpy.array or pd.series) and return true is classes is balanced
    def __is_balanced(self, y):
        class_count = Counter(y)
        total_samples = y.shape[0]
        ideal_count = total_samples/len(class_count)
        threshold = 0.20 * ideal_count
        return not any(abs(count - ideal_count) > threshold for count in class_count.values())
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this input ?
        Must be classification with balanced labels

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['binary', 'multiclass'] and not self.__is_balanced(y)
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return accuracy_score(y, y_pred) 
