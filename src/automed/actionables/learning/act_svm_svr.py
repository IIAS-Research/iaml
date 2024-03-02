from ...actionable import *
from ...output import Input

from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance


@isStep('learning', 'tabular')
@assessable
class ActSVMSVR(Actionable):
    name = "Learn : SVM Regression"
    def __init__(self):
        self.configurations = [{
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        }]
        
    @runner
    def run(self, input: Input, callback=None):
        self.model = svm.SVR(
            kernel = self.get_config('kernel')
            )
        
        if input.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input.dataset.X, input.dataset.y)
        
        return input.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
        
    
    def suitable(self, input: Input) -> bool:
        return input.dataset.type_of_target in ['continuous']
    
    def priorize(self, input=None):
        return 0.5 # neutral