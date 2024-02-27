from ...actionable import *
from ...automed import Output, Metric
from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance


def learn(model, X):
    return model.predict(X)


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
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metrics or Metric()
        
        model = svm.SVC(
            kernel = self.get_config('kernel'),
            random_state = self.get_config('random_state')
            )
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(model, learn)
    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['continuous']
    
    def priorize(self, input=None):
        return 0.5 # neutral