from ...actionable import *
from ...output import Input

from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance


@isStep('learning', 'tabular')
@assessable
class ActGaussianNb(Actionable):
    name = "Learn : Gaussian NB"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input: Input, callback=None):
        self.model = GaussianNB()
        
        if input.dataset.is_multilabel:
            self.model = BinaryRelevance(classifier=self.model, require_dense=[True, True])
        
        self.model.fit(input.dataset.X, input.dataset.y)
        
        return input.set_model(self)
    
    
    def learn(self, X):
        return self.model.predict(X)
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, input=None):
        return 0.5 # neutral