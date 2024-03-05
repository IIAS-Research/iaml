"""
[STEP] Learn :  XGBoost Regressor
"""
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from ...actionable import Actionable
from ...output import Input
from ...step import is_step, runner


@is_step('learning', 'tabular')
class ActXGBoost(Actionable):
    """
    [STEP] Learn :  XGBoost Regressor
    """
    name = "Learn : XGBoost Regressor"
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 1.0,
                'range': [0.000000001, float('inf')]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, float("inf")]
            }
        }
        self.model:GradientBoostingRegressor = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit XgBoost regressor on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = GradientBoostingRegressor(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
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
        return input_data.dataset.type_of_target in ['continuous']

    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
