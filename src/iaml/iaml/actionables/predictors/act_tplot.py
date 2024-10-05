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
    description = '''TPOTClassifier is a machine learning algorithm that uses genetic programming 
        to search for the best pipeline of machine learning models and hyperparameters for a given 
        classification task.'''
    description_long = ''' It works by generating a population of random pipelines, evaluating their
        performance using cross-validation, and then applying genetic operations such as mutation 
        and crossover to create new pipelines. The process is repeated until a stopping criterion 
        is met, such as a maximum number of generations or a maximum time limit.'''
    refs = [
        {
            'year': 2015,
            'name': 'TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning',
            'authors': [
                'Randal S. Olson',
                'Jason H. Moore'
            ],
            'doi': 'https://doi.org/10.1007/978-3-030-05318-5_8',
            'publisher': 'JMLR: Workshop and Conference Proceedings page 66--74'
        },
        {
            'year': 2016,
            'name': 'Evaluation of a Tree-based Pipeline Optimization Tool for \
                Automating Data Science',
            'authors': [
                'Randal S. Olson',
                'Nathan Bartley',
                'Ryan J. Urbanowicz'
                'Jason H. Moore'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1603.06212',
            'publisher': 'GECCO 2016'
        }
    ]
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
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
