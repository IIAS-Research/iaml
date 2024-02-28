from ...actionable import *
from ...automed import Output, Metric
from sklearn.neighbors import KNeighborsRegressor
from skmultilearn.problem_transform import BinaryRelevance


@isStep('learning', 'tabular', 'fast_learning')
@assessable
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
    def run(self, input:Output, callback=None):
        self.model = KNeighborsRegressor(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        self.model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(self)
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['continuous']

    def priorize(self, input=None):
        return 0.5 # neutral