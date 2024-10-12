"""
[METRIC] ROC AUC
"""
import textwrap
from sklearn.metrics import roc_auc_score
import pandas as pd
from ..metric import Metric

class RocAucMetric(Metric):
    """
    [METRIC] ROC AUC
    """

    name= "ROC AUC"
    description = textwrap.dedent('''\
        The ROC AUC metric (Receiver Operating Characteristic - Area Under the Curve) is a tool used to evaluate 
        the performance of a classification model. It measures the model's ability to distinguish between two 
        classes by indicating the proportion of true positives relative to false positives.''')
    description_long = textwrap.dedent('''\
        The ROC AUC metric is a method that helps assess the effectiveness of a classification model. 
        The ROC curve illustrates the model's performance at various decision thresholds, with a curve closer 
        to the top-left corner indicating better performance. The AUC, or "area under the curve," provides 
        a score between 0 and 1, where 1 means perfect predictions and 0.5 means the model is no better than 
        random guessing. In essence, the ROC AUC metric is a clear way to evaluate a 
        model's ability to correctly distinguish between two categories.''')
    refs=[
        {
            'year': 1982,
            'name': textwrap.dedent("""\
                The meaning and use of the area under a receiver operating characteristic (ROC) curve.
                """),
            'authors': [
                'Hanley James A.',
                'McNeil Barbara J.'
            ],
            'doi': 'https://doi.org/10.1148%2Fradiology.143.1.7063747',
            'publisher': 'Radiology. 143 (1): 29-36.'
        }
    ]

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
    
    @property
    def needed_prediction(self):
        return 'predict_proba'
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return roc_auc_score(y, y_pred[:, 1])
