from ...actionable import *
from ...automed import Output, Metric
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance


def learn(model, X):
    return model.predict(X)


@isStep('learning', 'tabular')
@assessable
class ActLogisticRegression(Actionable):
    name = "Learn : Logistic regression"
    def __init__(self):
        self.configurations = [{
            'max_iterations': {
                'description': 'Maximum number of iterations',
                'default': 1000,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metrics or Metric()
        
        model = LogisticRegression(
            max_iter = self.get_config('max_iterations'),
            n_jobs = -1,
            random_state = self.get_config('random_state')
            )
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(model, learn)
    
    def priorize(self, input=None):
        return 0.5 # neutral