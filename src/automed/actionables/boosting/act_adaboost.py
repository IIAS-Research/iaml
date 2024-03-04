from ...actionable import *
from ...automed import Metric
from ...output import Input

from sklearn.ensemble import AdaBoostClassifier

# TODO Adapt it two try all the compatible learning models

# @is_step('boosting', 'tabular')
class ActAdaBoost(Actionable):
    name = "Learn : AdaBoost"
    def __init__(self):
        self.configurations = [{
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'n_estimator': {
                'description': 'Number of estimators',
                'default': 2000
            },
        }]
        
    @runner
    def run(self, input_data: Input, callback=None):
        metric = input_data.metric or Metric()
        
        model = AdaBoostClassifier(
            input_data.model,
            n_estimators = self.get_config('n_estimator'),
            random_state= self.get_config('random_state')
            )
        
        model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.to_output(None, metric, model)

        
    
    def priorize(self, input_data=None):
        return 0.5 # neutral