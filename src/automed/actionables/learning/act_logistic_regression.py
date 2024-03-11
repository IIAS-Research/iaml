"""
[STEP] Learn :  Logistic Regression Classifier
"""
from sklearn.linear_model import LogisticRegression
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner


@is_step('learning', 'tabular', 'fast_learning')
class ActLogisticRegression(ActBaseLearning):
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
        
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
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
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)    
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
