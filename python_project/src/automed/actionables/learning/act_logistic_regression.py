from ...actionable import *
from ...automed import Output, Metric
from sklearn.linear_model import LogisticRegression

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
        metric = input.metric or Metric()
        
        model = LogisticRegression(
            max_iter = self.get_config('max_iterations'),
            n_jobs = -1,
            random_state = self.get_config('random_state')
            )
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral