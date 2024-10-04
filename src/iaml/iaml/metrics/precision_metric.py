"""
[METRIC] Precision
"""
import pandas as pd
from sklearn.metrics import precision_score
from ..metric import Metric
from ..type_of_target import type_of_target as get_type_of_target


class PrecisionMetric(Metric):
    """
    [METRIC] Precision
    """

    name="Precision"
    description = '''Precision measures the accuracy of positive predictions made by a model. 
        It indicates the proportion of true positive results among all positive predictions.'''
    description_long = '''Precision evaluates how many of the predicted positive cases are actually correct. 
        It is calculated as the number of true positives divided by the sum of true positives and false positives. 
        For example, if a model predicts 10 positive cases, and 7 of them are correct, the precision would be 70%. 
        This metric is important in healthcare to ensure that positive predictions are reliable, minimizing false alarms.'''
    refs=[
        {
            'year': 2007,
            'name': 'Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation',
            'authors': [
                'David M. W. Powers'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2010.16061',
            'publisher': 'Journal of Machine Learning Technologies'
        }
    ]

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
        if get_type_of_target(y) == 'binary':
            # TODO Find something less arbitrary
            return precision_score(y, y_pred, pos_label=y[0], zero_division=0.0)
        if get_type_of_target(y) == 'multiclass':
            return precision_score(y, y_pred, average = 'weighted', zero_division=0.0) 
        if get_type_of_target(y) == 'multilabel-indicator':
            return precision_score(y, y_pred, average= 'samples', zero_division=0.0)
        
        raise ValueError('Metric not suitable') 
