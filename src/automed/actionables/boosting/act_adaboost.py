"""
STEP
Apply AdaBoost on models
"""
from sklearn.ensemble import AdaBoostClassifier

from ...actionable import Actionable
from ...step import runner
from ...output import Input, Output


# @is_step('boosting', 'tabular')
class ActAdaBoost(Actionable):
    """
    Apply Adaboost on models
    
    Configuration:
        random_state: Random seed. Default 42
        n_estimator: Number of estimator. Default 2000
    """
    name:str = "Learn : AdaBoost"
    
    def __init__(self):
        self.configuration:dict = {
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'n_estimator': {
                'description': 'Number of estimators',
                'default': 2000
            },
        }
        
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """_summary_

        Args:
            input_data (Input): Input data
            callback (callable, optional): Call after each run. Defaults to None.

        Returns:
            Output: Transformed input
        """
        model:AdaBoostClassifier = AdaBoostClassifier(
            input_data.model,
            n_estimators = self.get_config('n_estimator'),
            random_state= self.get_config('random_state')
            )
        
        model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.to_output(None, None, model)

        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
