"""
[METRIC] Recall
"""
import textwrap
import pandas as pd
from sklearn.metrics import recall_score
from ..metric import Metric
from ..type_of_target import type_of_target  as get_type_of_target

class RecallMetric(Metric):
    """
    [METRIC] Recall
    """
    name= "Recall"
    description = textwrap.dedent('''\
        Recall measures the ability of a model to identify all relevant positive cases. 
        It indicates the proportion of true positive results among all actual positive cases.
        ''')
    description_long = textwrap.dedent('''\
        Recall evaluates how many actual positive cases were correctly predicted by the model. 
        It is calculated as the number of true positives divided by the total number of actual positives 
        (true positives + false negatives). For example, if there are 100 actual positive cases and the model 
        identifies 80, the recall would be 80 / (80 + 20) = 0.80 or 80%. 
        This metric is crucial in healthcare to ensure that positive cases are detected, reducing missed diagnoses.
        ''')
    
    refs=[
        {
            'year': 2007,
            'name': textwrap.dedent("""\
                Evaluation: From Precision, Recall and F-Measure to ROC,
                Informedness, Markedness & Correlation
                """),
            'authors': [
                'David M. W. Powers'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2010.16061',
            'publisher': 'Journal of Machine Learning Technologies'
        }
    ]
    
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
