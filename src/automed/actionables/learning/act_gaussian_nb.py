from ...actionable import *
from ...output import TrainingInput

from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance


def learn(model, X):
    return model.predict(X)


@isStep('learning', 'tabular')
@assessable
class ActGaussianNb(Actionable):
    name = "Learn : Gaussian NB"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input: TrainingInput, callback=None):
        model = GaussianNB()
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[True, True])
        
        model.fit(input.dataset.X_train, input.dataset.Y_train)
        
        return input.set_model(model, learn)
    

        
    
    def priorize(self, input=None):
        return 0.5 # neutral