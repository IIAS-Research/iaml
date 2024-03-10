"""
[STEP] Learn :  Logistic Regression Classifier
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...output import Input
from ...decorators.all import is_step


@is_step('learning', 'tabular', 'fast_learning')
class ActLogisticRegression(Actionable):
    """
    [STEP] Learn :  Logistic Regression Classifier
    """
    name = "Learn : Logistic Regression Classifier"
    def __init__(self):
        self.configuration:dict = {
            'max_iterations': {
                'description': 'Maximum number of iterations',
                'default': 1000,
                'range': [50, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:LogisticRegression = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit LOgistic Regression on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = LogisticRegression(
            max_iter = self.get_config('max_iterations'),
            n_jobs = -1,
            random_state = self.get_config('random_state')
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def predict(self, X:pd.DataFrame) -> list[float]:

        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
    
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
