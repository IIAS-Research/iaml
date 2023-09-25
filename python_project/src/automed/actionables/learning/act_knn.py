from ...actionable import *
from ...automed import Output, Metric
from sklearn.neighbors import KNeighborsClassifier

@isStep('learning', 'tabular')
@assessable
class ActKNN(Actionable):
    name = "Learn : KNN"
    def __init__(self):
        self.configurations = [{
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski'
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5
            }
        }]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metric or Metric()
        
        model = KNeighborsClassifier(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral