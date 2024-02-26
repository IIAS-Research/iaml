from ..step import *
from ..metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *
import numpy as np
 
class BalancedAccuracyMetric(Metric):
    
    def explain(self):
        return 'Compute the balanced accuracy.'
    
    def __str__(self):
        return 'balanced_accuracy'
    
    def suitable(self, y):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        for column in y.columns:
            if type_of_target(y[column]) in ['binary', 'multiclass']:
                return True
        
        return False
    
    def compute(self, y, y_pred):
        if type_of_target(y) in ['binary', 'multiclass']:
            return balanced_accuracy_score(y, y_pred) 
        elif type_of_target(y) in ['multilabel-indicator']:
            return np.mean([balanced_accuracy_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]) 