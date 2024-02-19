from ...step import *
from ...metric import *
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *

class F1ScoreMetric(Metric):
    
    def __str__(self):
        return 'f1_score'
    
    def explain(self):
        return 'Compute the F1 score, also known as balanced F-score or F-measure. The F1 score can be interpreted as a harmonic mean of the precision and recall'
    
    def _is_imbalanced(self, class_counts, total_samples):
        ideal_count = total_samples / len(class_counts)
        threshold = 0.15 * ideal_count
        return any(abs(count - ideal_count) > threshold for count in class_counts.values())
    
    def is_pertinent(self, y):
        #if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        #    y = y.iloc[:, 0]
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass', 'multilabel-indicator']:
            #class_count = Counter(y)
            #total_samples = sum(class_count.values())
            #relevance = self._is_imbalanced(class_count, total_samples)
            #if relevance:
            return True
            #else:
            #    return False
            
    def suitable(self, y):
        return self.is_pertinent(y)
    
    def compute(self, y, y_pred):
        if type_of_target(y) == 'binary':
            return f1_score(y, y_pred)
        elif type_of_target == 'multiclass':
            return f1_score(y, y_pred, average ='weighted')
        else:
            return f1_score(y, y_pred, average ='samples')