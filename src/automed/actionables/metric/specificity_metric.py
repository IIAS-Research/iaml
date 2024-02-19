from ...step import *
from ...metric import *
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix

class SpecificityMetric(Metric):
    
    def __str__(self):
        return 'specificity'
    
    def explain(self):
        return 'The proportion of negative instances that are corectely classified as negative: tn / (tn + fp)'
    
    #def _is_imbalanced(self, class_counts, total_samples):
    #    ideal_count = total_samples / len(class_counts)
    #    threshold = 0.15 * ideal_count
    #    return any(abs(count - ideal_count) > threshold for count in class_counts.values())
    
    def is_pertinent(self, y):
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        target_type = type_of_target(y)
        if target_type in ['binary']:
            #class_count = Counter(y)
            #total_samples = sum(class_count.values())
            #relevance = self._is_imbalanced(class_count, total_samples)
            #if relevance:
            return True
        else:
            return False
            
    def suitable(self, y):
        return self.is_pertinent(y)

    def compute(self, y, y_pred):
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0
        