from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_squared_log_error

class MeanSquaredLogErrorMetric(Metric):
    
    def __str__(self):
        return 'mean_squared_log_error'
    
    def explain(self):
        return 'Mean squared logarithmic error regression loss.'
    
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target == 'continuous' and not((dataset.y_train < 0).any())
    
    def compute(self, y, y_pred):
        try:
            return mean_squared_log_error(y, y_pred)
        except:
            return None