from ...actionable import *
from ...output import Input

from sklearn.ensemble import GradientBoostingRegressor


@is_step('learning', 'tabular')
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
    def run(self, input_data: Input, callback=None):
        self.model = GradientBoostingRegressor(
                                        n_estimators=self.get_config('n_estimators'),
                                        learning_rate=self.get_config('learning_rate'),
                                        max_depth=self.get_config('max_depth'),
                                        random_state=self.get_config('random_state')
                                        )
        
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def predict(self, X):
        return self.model.predict(X)

    
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['continuous']

    def priorize(self, input_data=None):
        return 0.5 # neutral