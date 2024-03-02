from ...actionable import *
from ...output import Input

from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance



@isStep('learning', 'tabular')
@assessable
class ActSVMSVC(Actionable):
    name = "Learn : SVM Classification"
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
    def run(self, input: Input, callback=None):
        self.model = svm.SVC(
            kernel = self.get_config('kernel'),
            class_weight = self.get_config('class_weight'),
            random_state = self.get_config('random_state'),
            probability = self.get_config('probability')
            )
        
        if input.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input.dataset.X, input.dataset.y)
        
        return input.set_model(self)
        
    def predict(self, X):
        return self.model.predict(X)
        
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input=None):
        return 0.5 # neutral