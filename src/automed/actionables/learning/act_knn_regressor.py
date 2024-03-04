from ...actionable import *
from ...output import Input

from sklearn.neighbors import KNeighborsRegressor


@is_step('learning', 'tabular', 'fast_learning')
class ActKNNRegressor(Actionable):
    name = "Learn : KNN"
    def __init__(self):
        self.configurations = [{
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski',
                'categorical': ['minkowski', 'manhattan']
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5,
                'range': [1, float('inf')]
            }
        }]
        
    @runner
    def run(self, input_data: Input, callback=None):
        self.model = KNeighborsRegressor(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['continuous']

    def priorize(self, input_data=None):
        return 0.5 # neutral