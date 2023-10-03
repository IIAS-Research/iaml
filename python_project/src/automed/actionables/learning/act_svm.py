from ...actionable import *
from ...automed import Output, Metric
from sklearn import svm

@isStep('learning', 'tabular')
@assessable
class ActSVM(Actionable):
    name = "Learn : SVM"
    def __init__(self):
        self.configurations = [{
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'probability': {
                'description': 'If true, the output will be a probability. If false, it will be Binary',
                'default': False
            },
            'class_weight': {
                'description': 'Can be set on "balenced" to improve results on unbalenced data',
                'default': None,
                'categorical': [None, {'balanced'}]
            }
        }]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metric or Metric()
        
        model = svm.SVC(
            kernel = self.get_config('kernel'),
            class_weight = self.get_config('class_weight'),
            random_state = self.get_config('random_state'),
            probability = self.get_config('probability')
            )
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral