"""
[METRIC] Mean Absolute Error
"""
import textwrap
import pandas as pd
from sklearn.metrics import mean_absolute_error
from ..metric import Metric

class MeanAbsoluteErrorMetric(Metric):
    """
    [METRIC] Mean Absolute Error
    """

    name = 'Mean Absolute Error'
    description = textwrap.dedent('''\
        Mean Absolute Error (MAE) evaluates the accuracy of a predictive model by calculating 
        the average of the absolute differences between predicted and actual values. It is a simple and intuitive metric 
        that helps understand how far off predictions are from the true outcomes.''')
    description_long = textwrap.dedent('''\
        Mean Absolute Error (MAE) evaluates a model's accuracy by calculating the average of the absolute differences 
        between predicted and actual values. It provides a clear understanding of how far predictions are from true outcomes. 
        To calculate MAE, you sum the absolute errors (the differences between predicted and actual values) and divide by the 
        total number of predictions. For example, if the errors are 2, -3, and 1, the MAE would be (|2| + |-3| + |1|) / 3 = 2. 
        This metric is valuable in healthcare for assessing the accuracy of continuous predictions, like estimating patient 
        outcomes.''')
    refs = [
        {
            'year': 2005,
            'name': textwrap.dedent("""\
                Advantages of the mean absolute error (MAE) over the root mean square error 
                (RMSE) in assessing average model performance"""),
            'authors': [
                'Willmott, Cort J',
                'Matsuura, Kenji'
            ],
            'doi': 'https://doi.org/10.3354%2Fcr030079',
            'publisher': ' Climate Research. 30: 79-82'
        }
    ]

    def __str__(self):
        return 'mean_absolute_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Mean absolute error regression loss.'
        
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
        return mean_absolute_error(y, y_pred)
