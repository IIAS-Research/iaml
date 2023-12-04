from ...actionable import *
from ...automed import Output, Metric
from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance

@isStep('learning', 'tabular')
@assessable
class ActSVM(Actionable):
    name = "Learn : SVM"
    def __init__(self):
        self.configurations = [{
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
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
                'categorical': [None, 'balanced']
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
        
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(model, lambda model, X: model.predict(X))
        

        
    
    def priorize(self, input=None):
        return 0.5 # neutral