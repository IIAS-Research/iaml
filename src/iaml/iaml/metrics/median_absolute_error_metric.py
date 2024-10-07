"""
[METRIC] Median Absolute Error
"""
import pandas as pd
from sklearn.metrics import median_absolute_error
from ..metric import Metric

class MedianAbsoluteErrorMetric(Metric):
    """
    [METRIC] Median Absolute Error
    """

    name="Median Absolute Error"
    description = '''Median Absolute Error (MedAE) is a metric used to evaluate the accuracy of a regression model. 
        It measures the median of the absolute differences between predicted values and actual values, providing a robust indication of prediction errors.'''
    description_long = '''Median Absolute Error (MedAE) is a metric that helps assess the accuracy of a regression 
        model by focusing on the errors in predictions. It calculates the absolute differences between the predicted values 
        and the actual values, then finds the median of these differences. This approach makes MedAE less sensitive to outliers 
        compared to other error metrics, as it focuses on the middle value of the errors. A lower MedAE indicates better model 
        performance, meaning the predictions are closer to the actual values. In summary, MedAE is a useful measure for 
        understanding the typical prediction error of a regression model.'''
    refs=[
        {
            'year': 1992,
            'name': 'Error measures for generalizing about forecasting methods: Empirical comparisons',
            'authors': [
                'Scott Armstrong', 
                'Fred Collopy'
            ],
            'doi': 'https://doi.org/10.1016/j.neucom.2015.12.114',
            'publisher': 'International Journal of Forecasting, Volume 8, Issue 1, June 1992, Pages 69-80'
        }
    ]
    def __str__(self):
        return 'median_absolute_error'
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Median absolute error regression loss.'
    
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
        return median_absolute_error(y, y_pred)
    