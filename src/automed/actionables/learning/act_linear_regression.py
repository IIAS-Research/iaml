from ...actionable import *
from ...automed import Output, Metric
from sklearn.linear_model import LinearRegression


def learn(model, X):
    return model.predict(X)


@isStep('learning', 'tabular', 'fast_learning')
@assessable
class ActLinearRegression(Actionable):
    name = "Learn : Linear Regression"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input:Output, callback=None):
        model = LinearRegression()
        model.fit(input.dataset.X_train, input.dataset.y_train)
        
        return input.set_model(model, learn)
    
    
    def suitable(self, input) -> bool:
        return input.dataset.type_of_target in ['continuous']
    
    def priorize(self, input=None):
        return 0.5 # neutral