from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_absolute_error

class MeanAbsoluteErrorMetric(Metric):
    
    def __str__(self):
        return 'mean_absolute_error'
    
    def explain(self):
        return 'Mean absolute error regression loss.'
        
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target == 'continuous'
    
    def compute(self, y, y_pred):
        return mean_absolute_error(y, y_pred)