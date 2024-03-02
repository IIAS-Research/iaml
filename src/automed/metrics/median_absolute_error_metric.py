from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import median_absolute_error

class MedianAbsoluteErrorMetric(Metric):
    
    def __str__(self):
        return 'median_absolute_error'
    
    def explain(self):
        return 'Median absolute error regression loss.'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target == 'continuous'
    
    def compute(self, y, y_pred):
        return median_absolute_error(y, y_pred)