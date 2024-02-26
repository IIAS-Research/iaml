from ...actionable import *
from ...output import TrainingInput

from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import BinaryRelevance


def learn(model, X):
    return model.predict(X)


@isStep('learning', 'tabular')
@assessable
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
    def run(self, input: TrainingInput, callback=None):
        model = KNeighborsClassifier(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.Y_train)
        
        return input.set_model(model, learn)
    

        
    
    def priorize(self, input=None):
        return 0.5 # neutral