"""[METRIC] R2 Score"""
from typing import Any
import textwrap
import pandas as pd
from sklearn.metrics import r2_score
from ..metric import Metric


class R2ScoreMetric(Metric):
    """[METRIC] R2 Score"""

    name: str = "R2 Score"
    _description: str = textwrap.dedent('''\
        R² Score, or R-squared, measures how well a model explains the variability of the target variable. 
        It indicates the proportion of variance in the data that is predictable from the model.''')
    _description_long: str = textwrap.dedent('''\
        R² Score evaluates the goodness of fit of a regression model. It ranges from 0 to 1, 
        where 1 means the model perfectly explains the variability of the target variable, and 0 means it does not 
        explain any variability. To calculate R², you compare the model's predictions to the mean of the actual values. 
        For example, an R² of 0.80 means that 80% of the variance in the target variable is explained by the model. 
        This metric is useful in healthcare to assess how well a model predicts outcomes.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1985,
            'name': 'Cautionary Note about R2',
            'authors': [
                'Robert G. D. Steel',
                'James H. Torrie'
            ],
            'doi': 'https://doi.org/10.2307/2287561',
            'publisher': textwrap.dedent("""\
                The American Statistician, Vol. 39, No. 4, Part 1 (Nov., 1985),
                wpp. 279-285 (7 pages)
                """)
        }
    ]

    def __str__(self) -> Any:
        return 'r2_score'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'continuous'

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        return r2_score(y, y_pred)
