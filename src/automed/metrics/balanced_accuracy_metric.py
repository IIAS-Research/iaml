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
    
    # Get one label (numpy.array or pd.series) and return false if classes is balanced
    def __is_balanced(self, y):
        class_count = Counter(y)
        total_samples = y.shape[0]
        ideal_count = total_samples/len(class_count)
        threshold = 0.20 * ideal_count
        return not(any(abs(count - ideal_count) > threshold for count in class_count.values()))
    
    def suitable(self, y):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        for column in y.columns:
            if not(type_of_target(y[column]) in ['binary', 'multiclass'] and self.__is_balanced(y[column])):
                return True
        
        return False
    
    def compute(self, y, y_pred):
        if type_of_target(y) in ['binary', 'multiclass']:
            return balanced_accuracy_score(y, y_pred) 
        elif type_of_target(y) in ['multilabel-indicator']:
            return np.mean([balanced_accuracy_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]) 