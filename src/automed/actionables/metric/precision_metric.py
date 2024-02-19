from ...step import *
from ...metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *
 
class precisionMetric(Metric):
    
    def __str__(self):
        return 'precision'
    
    def explain(self):
        return 'Compute the precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.'
    
    #def _is_imbalanced(self, class_counts, total_samples):
    #    ideal_count = total_samples/ len(class_counts)
    #    threshold = 0.20 * ideal_count
    #    return any(abs(count - ideal_count) > threshold for count in class_counts.vcalues())
    
    def is_pertinent(self, y):
        #if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        #    y = y.iloc[:,0]
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass', 'multilabel-indicator']:
            #class_count = Counter(y)
            #total_samples = sum(class_count.values())
            #is_pert = self._is_imbalanced(class_count, total_samples)
            return  True
        else:
            return False
        
    def suitable(self, y):
        return self.is_pertinent(y)
        
    
    def compute(self, y, y_pred):
        if type_of_target(y) == 'binary':
            return precision_score(y, y_pred)
        elif type_of_target(y) == 'multiclass':
            return precision_score(y, y_pred, average = 'weighted') 
        else:
            return precision_score(y, y_pred, average= 'samples')
           
