"""[METRIC] Classification Error"""
from typing import Any
import textwrap
import pandas as pd
from .balanced_accuracy_metric import BalancedAccuracyMetric
from ..metric import Metric


class ClassificationErrorMetric(Metric):
    """[METRIC] Classification Error"""
    name: str = 'Classification Error'
    description: str = textwrap.dedent('''\
        Classification Error measures a model's performance 
        by calculating the proportion of incorrect predictions. It is defined as 1 
        minus the Balanced Accuracy Score, making it useful for imbalanced datasets.
        ''')
    description_long: str = textwrap.dedent('''\
        Classification Error evaluates how well a predictive model performs 
        by measuring the proportion of incorrect predictions. It is calculated as 1 minus the Balanced 
        Accuracy Score, which gives equal importance to both positive and negative classes. 
        To calculate it, you first determine the Balanced Accuracy Score, which averages the accuracy 
        of both classes. Then, you subtract that value from 1. For example, if the Balanced Accuracy Score is 80%, 
        the Classification Error would be 1 - 0.80 = 0.20, or 20%. This metric helps highlight the model's shortcomings, 
        making it a valuable tool for assessing performance in medical decision-making.''')
    
    refs: list[dict[str, Any]] = [
        {
            'year': 2010,
            'name': 'The Balanced Accuracy and Its Posterior Distribution',
            'authors': [
                'Kay Henning Brodersen',
                'Cheng Soon Ong',
                'Klaas Enno Stephan',
                'Joachim M. Buhmann'
            ],
            'doi': 'https://doi.org/10.1109/ICPR.2010.764',
            'publisher': textwrap.dedent("""\
                Proceedings of the 20th International Conference on Pattern Recognition, 3121-24.
                """)
        },
        {
            'year': 2015,
            'name': textwrap.dedent("""\
                Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies
                """),
            'authors': [
                'John D. Kelleher',
                'Brian Mac Namee',
                'Aoife D\'Arcy'
            ],
            'doi': None,
            'publisher': textwrap.dedent("""\
                Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies
                """)
        }
    ]

    def __str__(self) -> str:
        return 'classification_error'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass', 'multilabel-indicator']

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        balanced_accuracy = BalancedAccuracyMetric().compute(y, y_pred)
        return 1 - balanced_accuracy
