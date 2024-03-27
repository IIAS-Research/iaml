"""
[STEP] Learn :  TPLOT
"""
from tpot import TPOTClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate

from ...decorators.all import is_step

# @is_step('predictor', 'tabular')
@is_step('to_compare')
class ActTPLOT(Predictor):
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
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit TPLOT on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
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

        self.model.fit(dataset.X, dataset.y)

        return self
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
