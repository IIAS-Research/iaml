"""
[STEP] Learn :  XGBoost
"""
from sklearn.ensemble import GradientBoostingClassifier
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActXGBoost(ActBaseLearning):
    """
    [STEP] Learn :  XGBoost
    """
    name = "Learn : XGBoost"
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
        self.model:GradientBoostingClassifier = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit XgBoost on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = GradientBoostingClassifier(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
                                        )
            
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
