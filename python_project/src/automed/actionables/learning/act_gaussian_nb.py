from ...actionable import *
from ...automed import Output, Metric


from sklearn.naive_bayes import GaussianNB

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
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.to_output(None, metric, model)

        
    
    def priorize(self, input=None):
        return 0.5 # neutral