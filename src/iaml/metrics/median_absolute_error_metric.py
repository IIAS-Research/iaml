"""[METRIC] Median Absolute Error"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import median_absolute_error
from ..metric import Metric


class MedianAbsoluteErrorMetric(Metric):
    """[METRIC] Median Absolute Error"""

    name: str = "Median Absolute Error"
    greater_is_better = False
    _description: str = textwrap.dedent('''\
        Median Absolute Error (MedAE) is a metric used to evaluate the accuracy of a regression model. 
        It measures the median of the absolute differences between predicted values and actual values, 
        providing a robust indication of prediction errors.
        ''')
    _description_long: str = textwrap.dedent('''\
        Median Absolute Error (MedAE) is a metric that helps assess the accuracy of a regression 
        model by focusing on the errors in predictions. It calculates the absolute differences between the predicted values 
        and the actual values, then finds the median of these differences. This approach makes MedAE less sensitive to outliers 
        compared to other error metrics, as it focuses on the middle value of the errors. A lower MedAE indicates better model 
        performance, meaning the predictions are closer to the actual values. In summary, MedAE is a useful measure for 
        understanding the typical prediction error of a regression model.
        ''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1992,
            'name': 'Error measures for generalizing about forecasting methods: ' \
                'Empirical comparisons',
            'authors': [
                'Scott Armstrong', 
                'Fred Collopy'
            ],
            'doi': 'https://doi.org/10.1016/j.neucom.2015.12.114',
            'publisher': 'International Journal of Forecasting, ' \
                'Volume 8, Issue 1, June 1992, Pages 69-80'
        }
    ]
    def __str__(self) -> str:
        return 'median_absolute_error'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'continuous'

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        return median_absolute_error(y, y_pred)
