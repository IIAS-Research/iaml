from ...actionable import *
from ...output import Input

from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance


@is_step('learning', 'tabular')
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
    def run(self, input_data: Input, callback=None):
        self.model = svm.SVR(
            kernel = self.get_config('kernel')
            )
        
        if input_data.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[False, True])
        
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
        
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['continuous']
    
    def priorize(self, input_data=None):
        return 0.5 # neutral