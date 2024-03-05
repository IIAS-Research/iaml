"""
[STEP] Learn : KNN
"""
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from ...actionable import Actionable
from ...output import Input
from ...step import is_step, runner



@is_step('learning', 'tabular', 'fast_learning')
class ActKNNRegressor(Actionable):
    """
    [STEP] Learn : KNN
    """
    name = "Learn : KNN"
    def __init__(self):
        self.configuration:dict = {
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski',
                'categorical': ['minkowski', 'manhattan']
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5,
                'range': [1, float('inf')]
            }
        }
        self.model:KNeighborsRegressor = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit Knn regressor on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = KNeighborsRegressor(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def predict(self, X:pd.DataFrame) -> list[float]:

        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
    
    
    def suitable(self, input_data: Input) -> bool:
        """
        Does this step suitable for this input

        Args:
            input_data (Input): Suitable for this input

        Returns:
            bool: Suitable ?
        """
        return input_data.dataset.type_of_target in ['continuous']

    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
