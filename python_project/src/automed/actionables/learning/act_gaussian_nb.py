from ...actionable import *
from ...automed import Output, Metric


from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance

@isStep('learning', 'tabular')
@assessable
class ActGaussianNb(Actionable):
    name = "Learn : Gaussian NB"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input:Output, callback=None):
        metric = input.metric or Metric()
        
        model = GaussianNB()
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[True, True])
        
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral
    