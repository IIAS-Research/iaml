"""
[METRIC] Mean Squared Error
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from ..metric import Metric

class MeanSquaredErrorMetric(Metric):
    """
    [METRIC] Mean Squared Error
    """

    name = 'Mean Squared Error'
    description = '''Mean Squared Error (MSE) measures the average of the squares of the errors 
        in predictions. It emphasizes larger errors, making it useful for assessing prediction accuracy.'''
    description_long = '''Mean Squared Error (MSE) evaluates a model's accuracy by calculating the average of the squared 
        differences between predicted and actual values. It gives more weight to larger errors, which is important in healthcare.
        To calculate MSE, you square each error, sum them, and divide by the total number of predictions. 
        For example, if the errors are 2, -3, and 1, the MSE would be (2² + (-3)² + 1²) / 3 = 4.67. 
        This metric helps identify how well a model performs, especially when larger errors matter more.'''
    refs = [
        {
            'year': 2006,
            'name': 'Mathematical Statistics: Basic Ideas and Selected Topics',
            'authors': [
                'Bickel, Peter J'
            ],
            'doi': 'https://doi.org/10.1201/9781315369266',
            'publisher': ' Mathematical Statistics: Basic Ideas and Selected Topics. Vol. I (Second ed.). p. 20'
        }
    ]

    def __str__(self):
        return 'mean_squared_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Mean squared error regression loss.'
        
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
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame) -> float:
        """
        Compute metric with predicted data

        Args:
            y (pd.DataFrame): Ground truth data
            y_pred (pd.DataFrame): Predicted data

        Returns:
            float: computed value 
        """
        return mean_squared_error(y, y_pred)
