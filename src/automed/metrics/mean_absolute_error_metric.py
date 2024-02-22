from ..step import *
from ..metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_absolute_error

class MeanAbsoluteErrorMetric(Metric):
    
    def __str__(self):
        return 'mean_absolute_error'
    
    def explain(self):
        return 'Mean absolute error regression loss.'
        
    def suitable(self, y):
        target_type = type_of_target(y)
        if target_type == 'continuous':
            return True
        else:
            return False
    
    def compute(self, y, y_pred):
        return mean_absolute_error(y, y_pred)