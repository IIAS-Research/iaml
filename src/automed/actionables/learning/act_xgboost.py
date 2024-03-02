from ...actionable import *
from ...output import Input

from sklearn.ensemble import GradientBoostingClassifier
from skmultilearn.problem_transform import BinaryRelevance


@isStep('learning', 'tabular')
@assessable
class ActXGBoost(Actionable):
    name = "Learn : XGBoost"
    def __init__(self):
        self.configurations = [{
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
        }]
        
    @runner
    def run(self, input: Input, callback=None):
        self.model = GradientBoostingClassifier(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
                                        )
        
        
        if input.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
            
            
        self.model.fit(input.dataset.X, input.dataset.y)
        
        return input.set_model(self)
    
    def predict(self, X):
        return self.model.predict(X)

    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, input=None):
        return 0.5 # neutral