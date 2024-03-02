from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_squared_error

class MeanSquaredErrorMetric(Metric):
    
    def __str__(self):
        return 'mean_squared_error'
    
    def explain(self):
        return 'Mean squared error regression loss.'
        
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target == 'continuous'
    
    def compute(self, y, y_pred):
        return mean_squared_error(y, y_pred)