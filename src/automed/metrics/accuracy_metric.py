from ..step import *
from ..metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *
 
class AccuracyMetric(Metric):
    
    def __str__(self):
        return 'accuracy'
    
    def explain(self):
        return 'Accuracy classification score.'
    
    # Get one label (numpy.array or pd.series) and return true is classes is balanced
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
                return False
        
        return True
         
    def compute(self, y, y_pred):
        return accuracy_score(y, y_pred) 