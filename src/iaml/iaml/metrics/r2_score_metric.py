"""
[METRIC] R2 Score
"""
import textwrap
import pandas as pd
from sklearn.metrics import r2_score
from ..metric import Metric

class R2ScoreMetric(Metric):
    """
    [METRIC] R2 Score
    """

    name = "R2 Score"
    description = textwrap.dedent('''\
        R² Score, or R-squared, measures how well a model explains the variability of the target variable. 
        It indicates the proportion of variance in the data that is predictable from the model.''')
    description_long = textwrap.dedent('''\
        R² Score evaluates the goodness of fit of a regression model. It ranges from 0 to 1, 
        where 1 means the model perfectly explains the variability of the target variable, and 0 means it does not 
        explain any variability. To calculate R², you compare the model's predictions to the mean of the actual values. 
        For example, an R² of 0.80 means that 80% of the variance in the target variable is explained by the model. 
        This metric is useful in healthcare to assess how well a model predicts outcomes.''')
    refs=[
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

    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'R2(coefficient of determination) regression score function. \
            Best possible score is 1.0 and it can be negative \
            (because the model can be arbitrarily worse).'
    
    def __str__(self):
        return 'r2_score'
        
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        """
        Does this metric is suitable for this candidate ?
        Must be regression

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels
            type_of_target (str): Type of target

        Returns:
            bool: Suitable ?
        """
        return type_of_target == 'continuous'

    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return r2_score(y, y_pred)
