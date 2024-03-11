"""
[STEP] Learn :  Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner

@is_step('learning', 'tabular')
class ActRandomForest(ActBaseLearning):
    """
    [STEP] Learn :  Random Forest
    """
    name = "Learn : Random Forest" 
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:RandomForestClassifier = None
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit Random forest on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.model = RandomForestClassifier(max_depth=self.get_config('max_depth'),
                                            random_state=self.get_config('random_state'),
                                            n_estimators=self.get_config('n_estimators'))
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
