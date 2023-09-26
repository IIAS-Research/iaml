from ..actionable import *
from ..automed import Output
import numpy as np
from sklearn.model_selection import train_test_split


@isStep('split')
class RandomSplit(Actionable):
    name = "Split date to train and test set"
    def __init__(self):
        self.configurations = [{
            'ratio': {
                'description': 'Split ratio',
                'default': 0.2
            },
            'random_state': {
                'description': 'Random state',
                'default': 42
            }
        }]
        
    @runner
    def run(self, input, callback=None) -> Output:
        X_train, X_test, y_train, y_test = train_test_split(
            input.dataset.X_train,
            input.dataset.y_train,
            test_size=self.get_config('ratio'),
            random_state=self.get_config('random_state')
        )
        
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        
        input.dataset.split(X_train, y_train, X_test, y_test)
        
        return input.to_output(input.dataset, None, None)
        
    
    def priorize(self, input=None):
        return 1