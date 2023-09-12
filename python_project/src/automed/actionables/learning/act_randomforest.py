from ...actionable import *
from ...automed import Output, Metric
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

@isStep('learning', 'tabular')
@assessable
class ActRandomForest(Actionable):
    name = "Learn : Random Forest"
    configuration = {
        'max_depth': {
            'description': 'Max depth of each tree',
            'default': 15
        },
        'random_state': {
            'description': 'random_state',
            'default': 42
        }
    }
    
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metric or Metric()
        
        model = RandomForestClassifier(max_depth=self.get_config('max_depth'), random_state=self.get_config('random_state'))
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral