"""[METRIC] Mean Squared Error"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import mean_squared_error
from ..metric import Metric


class MeanSquaredErrorMetric(Metric):
    """[METRIC] Mean Squared Error"""

    name: str = 'Mean Squared Error'
    description: str = textwrap.dedent('''\
        Mean Squared Error (MSE) measures the average of the squares of the errors 
        in predictions. It emphasizes larger errors, making it useful for assessing prediction accuracy.
        ''')
    description_long: str = textwrap.dedent('''\
        Mean Squared Error (MSE) evaluates a model's accuracy by calculating the average of the squared 
        differences between predicted and actual values. It gives more weight to larger errors, which is important in healthcare.
        To calculate MSE, you square each error, sum them, and divide by the total number of predictions. 
        For example, if the errors are 2, -3, and 1, the MSE would be (2² + (-3)² + 1²) / 3 = 4.67. 
        This metric helps identify how well a model performs, especially when larger errors matter more.
        ''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Mathematical Statistics: Basic Ideas and Selected Topics',
            'authors': [
                'Bickel, Peter J'
            ],
            'doi': 'https://doi.org/10.1201/9781315369266',
            'publisher': textwrap.dedent("""\
                Mathematical Statistics: Basic Ideas and Selected Topics. Vol. I (Second ed.). p. 20
                """)
        }
    ]

    def __str__(self) -> str:
        return 'mean_squared_error'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'continuous'

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        return mean_squared_error(y, y_pred)
