from ...actionable import *
from ...output import Input

from sklearn.linear_model import LinearRegression


@is_step('learning', 'tabular', 'fast_learning')
class ActLinearRegression(Actionable):
    name = "Learn : Linear Regression"
    def __init__(self):
        self.configurations = [{}]
        
    @runner
    def run(self, input_data: Input, callback=None):
        self.model = LinearRegression()
        self.model.fit(input_data.dataset.X, input_data.dataset.y)
        
        return input_data.set_model(self)
    
    
    def predict(self, X):
        return self.model.predict(X)
    
    def suitable(self, input_data: Input) -> bool:
        return input_data.dataset.type_of_target in ['continuous']
    
    def priorize(self, input_data=None):
        return 0.5 # neutral