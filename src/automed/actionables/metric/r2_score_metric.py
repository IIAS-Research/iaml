from ... step import *
from ...metric import *
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import r2_score

class R2ScoreMetric(Metric):
    
    def explain(self):
        return 'R2(coefficient of determination) regression score function. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).'
    
    def __str__(self):
        return 'r2_score'
    
    def is_pertinent(self, y):
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        target_type = type_of_target(y)
        if target_type == 'continuous':
            return True
        else:
            return False
        
    def suitable(self, y):
        return self.is_pertinent(y)
    
    def compute(self, y, y_pred):
        return r2_score(y, y_pred)
            
