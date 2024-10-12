"""
[METRIC] F1 Score
"""
import textwrap
import pandas as pd
from sklearn.metrics import f1_score
from ..metric import Metric
from ..type_of_target import type_of_target as get_type_of_target

class F1ScoreMetric(Metric):
    """
    [METRIC] F1 Score
    """
    name = 'F1 Score'
    description = textwrap.dedent('''\
        F1 Score is a metric that combines precision and recall to 
        evaluate a model's performance. It is especially useful for imbalanced datasets, 
        providing a balance between false positives and false negatives.''')
    description_long = textwrap.dedent('''\
        F1 Score measures a model's performance by combining precision 
        (the accuracy of positive predictions) and recall (the ability to identify all positive cases). 
        It is calculated as the harmonic mean of precision and recall, making it particularly useful in 
        healthcare when dealing with imbalanced data.
        The F1 Score ranges from 0 to 1, where 1 indicates perfect precision and recall. 
        For example, if a model has a precision of 70% and a recall of 80%, the F1 Score would be 
        calculated as 2 * (0.70 * 0.80) / (0.70 + 0.80) = 0.74. This metric helps ensure that both false positives 
        and false negatives are considered, making it valuable for medical decision-making.''')
    
    refs = [
        {
            'year': 2007,
            'name': 'The truth of the F-measure',
            'authors': [
                'Yutaka Sasaki'
            ],
            'doi': 'https://www.researchgate.net/publication/268185911_The_truth_of_the_F-measure',
            'publisher': ' Teach Tutor Mater. Vol. 1, no. 5. pp. 1–5'
        }
    ]

    def __str__(self):
        return 'f1_score'
            
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
            # TODO Find something less arbitrary (about pos_label)
            return f1_score(y, y_pred, pos_label=y[0])
        if get_type_of_target(y) == 'multiclass':
            return f1_score(y, y_pred, average ='weighted')
        if get_type_of_target(y) == 'multilabel-indicator':
            return f1_score(y, y_pred, average ='samples')
        
        raise ValueError('Metric not suitable') 
