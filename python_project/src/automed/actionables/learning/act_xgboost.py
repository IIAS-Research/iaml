from ...actionable import *
from ...automed import Output, Metric
from sklearn.ensemble import GradientBoostingClassifier

@isStep('learning', 'tabular')
@assessable
class ActXGBoost(Actionable):
    name = "Learn : XGBoost"
    def __init__(self):
        self.configurations = [{
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 1.0
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100
            }
        },
        {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 2
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 1
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 1
            }
        }]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metric or Metric()
        
        model = GradientBoostingClassifier(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
                                        )
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral