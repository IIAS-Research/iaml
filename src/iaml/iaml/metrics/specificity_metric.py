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

    name="Specificity"
    description = '''Specificity is a metric used to evaluate the performance of a classification model. It measures the proportion of true negatives 
        correctly identified by the model, indicating how well it can identify the negative class.'''
    description_long = '''Specificity is a metric that helps assess how well a classification model identifies the negative class. 
        For example, if you're predicting whether a medical test result is negative for a disease, specificity tells you the 
        percentage of actual negative cases that the model correctly identifies. A high specificity means the model is good at 
        avoiding false positives, while a low specificity indicates it may incorrectly label negative cases as positive.'''
    refs=[
        {
            'year': 1994 ,
            'name': 'Diagnostic tests. 1: Sensitivity and specificity.',
            'authors': [
                'D. G. Altman',
                'J. M. Bland'
            ],
            'doi': 'https://doi.org/10.1136%2Fbmj.308.6943.1552',
            'publisher': 'BMJ. 308 (6943): 1552'
        }
    ]
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
