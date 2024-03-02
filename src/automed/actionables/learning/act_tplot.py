from ...actionable import *
from ...output import Input

from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier

# @isStep('learning', 'tabular')
@isStep('to_compare')
@assessable
class ActTPLOT(Actionable):
    name="Learn : TPLOT"
    def __init__(self):
        self.configurations = [{
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
        }]
    
    @runner
    def run(self, input: Input, callback=None):
        
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

        self.model.fit(input.dataset.X, input.dataset.y)

        return input.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)

    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator', 'continuous']

    def priorize(self, input=None):
        return 0.5 # neutral