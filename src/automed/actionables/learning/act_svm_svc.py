from ...actionable import *
from ...output import Input

from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance



@is_step('learning', 'tabular')
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
    def run(self, input_data: Input, callback=None):
        self.model = svm.SVC(
            kernel = self.get_config('kernel'),
            class_weight = self.get_config('class_weight'),
            random_state = self.get_config('random_state'),
            probability = self.get_config('probability')
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