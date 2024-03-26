"""
[METRIC] Specificity Multilabel
"""
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from ..metric import Metric

class SpecificityMultilabelMetric(Metric):
    """
    [METRIC] Specificity Multilabel
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Compute specificity for each label separately, \
            then averaged to obtain an overall measure.'
    
    def _str__(self):
        return 'specificity_multilabel'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be multilabel classification 

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target in ['multilabel-indicator']
    
    # Specificity is calculated for each label separately, 
    # then averaged to obtain an overall measure. 
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        # Generates a series of confusion matrices,  one for each label
        mcm = multilabel_confusion_matrix(y, y_pred)
        specificity_per_label = []
        for i in range(mcm.shape[0]):
            tn, fp, _, _ = mcm[i].ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            specificity_per_label.append(specificity)
                
        mean_specificity = np.mean(specificity_per_label)
        return mean_specificity
