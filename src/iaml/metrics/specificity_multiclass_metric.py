"""[METRIC] Specificity Multiclass"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from ..metric import Metric


class SpecificityMulticlassMetric(Metric):
    """[METRIC] Specificity Multiclass"""

    name: str = "Specificity Multiclass"
    _description: str = textwrap.dedent('''\
        Multiclass specificity is a metric used to evaluate the performance of a classification model
        with multiple classes. It measures how well the model identifies the negative cases for each
        class by considering true negatives and false positives.''')
    _description_long: str = textwrap.dedent('''\
        Multiclass specificity assesses how effectively a classification model identifies negative cases
        across multiple classes. For each class, it calculates the number of true negatives (correctly 
        identified negatives) and false positives (incorrectly identified positives). 
        By summing these values for all classes, you can determine an overall specificity score. 
        A high multiclass specificity indicates that the model is good at correctly identifying non-target classes, 
        while a low score suggests it may struggle with misclassifying negative cases.''')
    refs: list[dict[str, Any]] = [
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

    def __str__(self) -> str:
        return 'specificity_multiclass'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'multiclass'

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        # Specificity is calculated by summing the true negartives and false
        # positives for each class, then using these totals to obtain an overall specificity"
        cm = confusion_matrix(y, y_pred)
        total_tn = 0
        total_fp = 0
        for i in range(len(cm)):
            total_tn += np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            total_fp += np.sum(cm[:, i]) - cm[i, i]

        return total_tn / (total_tn + total_fp)
