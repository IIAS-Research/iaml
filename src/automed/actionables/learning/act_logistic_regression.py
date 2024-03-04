from ...actionable import *
from ...output import Input

from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance


@is_step('learning', 'tabular', 'fast_learning')
class ActLogisticRegression(Actionable):
    name = "Learn : Logistic Regression Classifier"
    def __init__(self):
        self.configurations = [{
            'max_iterations': {
                'description': 'Maximum number of iterations',
                'default': 1000,
                'range': [50, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }]
        
    @runner
    def run(self, input_data: Input, callback=None):
        self.model = LogisticRegression(
            max_iter = self.get_config('max_iterations'),
            n_jobs = -1,
            random_state = self.get_config('random_state')
            )
        
        if input_data.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input_data=None):
        return 0.5 # neutral