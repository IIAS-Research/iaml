from ...actionable import *
from ...automed import Output, Metric
from sklearn.linear_model import LinearRegression


@isStep('learning', 'tabular', 'fast_learning')
@assessable
class ActLinearRegression(Actionable):
    name = "Learn : Linear Regression"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input:Output, callback=None):
        self.model = LinearRegression()
        self.model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['continuous']
    
    def priorize(self, input=None):
        return 0.5 # neutral