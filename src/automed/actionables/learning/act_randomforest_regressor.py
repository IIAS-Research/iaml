from ...actionable import *
from ...automed import Output, Metric
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestRegressor


@isStep('learning', 'tabular')
@assessable
class ActRandomForestRegressor(Actionable):
    name = "Learn : Random Forest Regressor" 
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
        self.model = RandomForestRegressor(
            max_depth=self.get_config('max_depth'),
            random_state=self.get_config('random_state'),
            n_estimators=self.get_config('n_estimators')
            )
        
        if input.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['continuous']

    def priorize(self, input=None):
        return 0.5 # neutral