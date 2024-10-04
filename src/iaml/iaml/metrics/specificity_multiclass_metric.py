"""
[METRIC] Specificity Multiclass
"""
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from ..metric import Metric


class SpecificityMulticlassMetric(Metric):
    """
    [METRIC] Specificity Multiclass
    """

    name="Specificity Multiclass"
    description = '''Multiclass specificity is a metric used to evaluate the performance of a classification model with multiple classes. 
        It measures how well the model identifies the negative cases for each class by considering true negatives and false positives.'''
    description_long = '''Multiclass specificity assesses how effectively a classification model identifies negative cases across multiple classes. 
        For each class, it calculates the number of true negatives (correctly identified negatives) and false positives (incorrectly identified positives). 
        By summing these values for all classes, you can determine an overall specificity score. 
        A high multiclass specificity indicates that the model is good at correctly identifying non-target classes, 
        while a low score suggests it may struggle with misclassifying negative cases.'''
    refs=[
        {
            'year': 2020,
            'name': 'Metrics for Multi-Class Classification: an Overview',
            'authors': [
                'Margherita Grandini',
                'Enrico Bagli',
                'Giorgio Visani'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2008.05756',
            'publisher': ''
        }
    ]

    def __str__(self):
        return 'specificity_multiclass'    
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Calculate specificity in a multiclass conext, where each instance belongs \
            to just one of several classes'
    
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
        return type_of_target in ['multiclass']
    
    
    # Specificity is calculated by summing the true negartives and false 
    # positives for each class, then using these totals to obtain an overall specificity"
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        cm = confusion_matrix(y, y_pred)
        total_tn = 0
        total_fp = 0
        for i in range(len(cm)):
            total_tn += np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            total_fp += np.sum(cm[:, i]) - cm[i, i]
        return total_tn / (total_tn + total_fp)
