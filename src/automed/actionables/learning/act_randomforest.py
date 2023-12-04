from ...actionable import *
from ...automed import Output, Metric
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skmultilearn.problem_transform import BinaryRelevance

@isStep('learning', 'tabular')
@assessable
class ActRandomForest(Actionable):
    name = "Learn : Random Forest" 
    def __init__(self):
        self.configurations = [{
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
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
        
        model = RandomForestClassifier(max_depth=self.get_config('max_depth'), random_state=self.get_config('random_state'), n_estimators=self.get_config('n_estimators'))
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(model, lambda model, X: model.predict(X))

        
    
    def priorize(self, input=None):
        return 0.5 # neutral