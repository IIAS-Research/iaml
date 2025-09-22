"""[METRIC] Recall"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import recall_score
from ..metric import Metric
from ..type_of_target import type_of_target  as get_type_of_target


class RecallMetric(Metric):
    """[METRIC] Recall"""

    name: str = "Recall"
    _description: str = textwrap.dedent('''\
        Recall measures the ability of a model to identify all relevant positive cases. 
        It indicates the proportion of true positive results among all actual positive cases.
        ''')
    _description_long: str = textwrap.dedent('''\
        Recall evaluates how many actual positive cases were correctly predicted by the model. 
        It is calculated as the number of true positives divided by the total number of actual positives 
        (true positives + false negatives). For example, if there are 100 actual positive cases and the model 
        identifies 80, the recall would be 80 / (80 + 20) = 0.80 or 80%. 
        This metric is crucial in healthcare to ensure that positive cases are detected, reducing missed diagnoses.
        ''')
    refs: list[dict[str, Any]] = [
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

    def __str__(self) -> str:
        return 'recall'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        match get_type_of_target(y):
            case 'multiclass':
                return recall_score(y, y_pred, average = 'weighted', zero_division=0.0)
            case 'multilabel-indicator':
                return recall_score(y, y_pred, average= 'samples', zero_division=0.0)
            case _:
                # Binary case otherwise
                # TODO Find something less arbitrary
                return recall_score(y, y_pred, pos_label=y[0], zero_division=0.0)
