"""[METRIC] Specificity"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import confusion_matrix
from ..metric import Metric

class SpecificityMetric(Metric):
    """[METRIC] Specificity"""

    name: str = "Specificity"
    description: str = textwrap.dedent('''\
        Specificity is a metric used to evaluate the performance of a classification model. 
        It measures the proportion of true negatives correctly identified by the model, indicating 
        how well it can identify the negative class.''')
    description_long: str = textwrap.dedent('''\
        Specificity is a metric that helps assess how well a classification model identifies the negative class. 
        For example, if you're predicting whether a medical test result is negative for a disease, specificity tells you the 
        percentage of actual negative cases that the model correctly identifies. A high specificity means the model is good at 
        avoiding false positives, while a low specificity indicates it may incorrectly label negative cases as positive.
        ''')
    refs: list[dict[str, Any]]=[
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

    def __str__(self) -> str:
        return 'specificity'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target in ['binary']

    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0
