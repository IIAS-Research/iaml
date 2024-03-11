"""
[STEP] Learn :  TPLOT
"""
from tpot import TPOTClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from .act_base_learning import ActBaseLearning
from ...output import Input
from ...step import is_step, runner

# @is_step('learning', 'tabular')
@is_step('to_compare')
class ActTPLOT(ActBaseLearning):
    """
    [STEP] Learn :  TPLOT
    """
    name="Learn : TPLOT"
    def __init__(self):
        self.configuration:dict = {
            'random_state': {
                'description': 'Random state TODO',
                'default': 42
            },
            'n_repeats': {
                'description': 'n repeats TODO',
                'default': 3
            },
            'n_splits': {
                'description': 'n splits TODO',
                'default': 10
            },
            'generations': {
                'description': 'generations TODO',
                'default': 5
            },
            'population_size': {
                'description': 'population size TODO',
                'default': 50
            },
            'n_jobs': {
                'description': 'n jobs TODO',
                'default': -1
            },
        }
        self.model: TPOTClassifier = None
    
    @runner
    def run(self, input_data: Input, callback=None): # pylint: disable=unused-argument
        """
        Fit TPLOT on Input.dataset

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        
        cv = RepeatedStratifiedKFold(
            n_splits = self.get_config('n_splits'),
            n_repeats = self.get_config('n_repeats'),
            random_state = self.get_config('random_state')
            )
        
        self.model = TPOTClassifier(
            generations = self.get_config('generations'),
            population_size = self.get_config('population_size'),
            cv=cv,
            scoring='accuracy',
            verbosity=2,
            random_state = self.get_config('random_state'),
            n_jobs = self.get_config('n_jobs')
            )

        self.model.fit(input_data.dataset.X, input_data.dataset.y)

        return input_data.set_model(self)
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
