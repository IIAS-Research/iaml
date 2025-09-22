"""[METRIC] Mean Squared Log Error"""
from typing import Any
import textwrap
from sklearn.metrics import mean_squared_log_error
import pandas as pd
from ..metric import Metric


class MeanSquaredLogErrorMetric(Metric):
    """[METRIC] Mean Squared Log Error"""
    name: str = 'Mean Squared Log Error'
    _description: str = textwrap.dedent('''\
        Mean Squared Log Error (MSLE) measures the average of the squared differences 
        between the logarithm of predicted and actual values. It is useful for data with wide-ranging values.
        ''')
    _description_long: str = textwrap.dedent('''\
        Mean Squared Log Error (MSLE) evaluates a model's accuracy by calculating the 
        average of the squared differences between the logarithms of predicted and actual values. 
        This metric is helpful when the target variable varies greatly in scale. 
        To calculate MSLE, you take the logarithm of both predicted and actual values, find the differences, 
        square them, and then average these squared differences. This approach reduces the impact of large errors, 
        making it valuable for assessing model performance in cases where relative differences matter more than 
        absolute differences.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2021,
            'name': 'Mean Squared Error, Deconstructed',
            'authors': [
                'Timothy O. Hodson',
                'Thomas M. Over',
                'Sydney Foks'    
            ],
            'doi': 'http://dx.doi.org/10.1029/2021MS002681',
            'publisher': ' Journal of Advances in Modeling Earth Systems, volume 13'
        }
    ]

    def __str__(self) -> str:
        return 'mean_squared_log_error'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'continuous' and not (y < 0).any(axis=None)

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float | None:
        try:
            return mean_squared_log_error(y, y_pred)
        except:  # pylint: disable=bare-except
            return None
