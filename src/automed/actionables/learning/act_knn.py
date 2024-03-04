from ...actionable import *
from ...output import Input

from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import BinaryRelevance


@is_step('learning', 'tabular')
class ActKNN(Actionable):
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
        self.model = KNeighborsClassifier(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        if input_data.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def suitable(self, input_data) -> bool:
        return input_data.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, input_data=None):
        return 0.5 # neutral