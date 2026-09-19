"""[METRIC] Precision"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.utils.multiclass import type_of_target as get_type_of_target
from ._classification import resolve_pos_label
from ..metric import Metric


class PrecisionMetric(Metric):
    """[METRIC] Precision.

    :param pos_label: Binary positive class. If None, use 1 for 0/1 or -1/1
        labels, otherwise the last sorted class. Training labels passed to
        compute as y_train take precedence for automatic class selection.
        Ignored for multiclass and multilabel targets.
    """

    name: str = "Precision"
    _description: str = textwrap.dedent('''\
        Precision measures the accuracy of positive predictions made by a model. 
        It indicates the proportion of true positive results among all positive predictions.
        ''')
    _description_long: str = textwrap.dedent('''\
        Precision evaluates how many of the predicted positive cases are actually correct. 
        It is calculated as the number of true positives divided by the sum of true positives and false positives. 
        For example, if a model predicts 10 positive cases, and 7 of them are correct, the precision would be 70%. 
        This metric is important in healthcare to ensure that positive predictions are reliable, minimizing false alarms.
        ''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2007,
            'name': textwrap.dedent("""\
                Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation
                """),
            'authors': [
                'David M. W. Powers'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2010.16061',
            'publisher': 'Journal of Machine Learning Technologies'
        }
    ]

    def __init__(self, pos_label: Any = None) -> None:
        self.pos_label = pos_label

    def __str__(self) -> str:
        return 'precision'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass',  'multilabel-indicator']

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        y_train = kwargs.get('y_train')
        match get_type_of_target(y if y_train is None else y_train):
            case 'multiclass':
                return precision_score(y, y_pred, average = 'weighted', zero_division=0.0)
            case 'multilabel-indicator':
                return precision_score(y, y_pred, average= 'samples', zero_division=0.0)
            case _:
                pos_label = resolve_pos_label(y, self.pos_label, y_train)
                return precision_score(y, y_pred, pos_label=pos_label, zero_division=0.0)
