from ...step import *
from ...metric import *
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

class SpecificityMulticlassMetric(Metric):
    
    def __str__(self):
        return 'specificity'    
    
    def explain(self):
        return 'Calculate specificity in a multiclass conext, where each instance belongs to just one of several classes'
    
    #def _is_imbalanced(self, class_counts, total_samples):
    #    ideal_count = total_samples / len(class_counts)
    #    threshold = 0.15 * ideal_count
    #    return any(abs(count - ideal_count) > threshold for count in class_counts.values())
    
    def is_pertinent(self, y):
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        target_type = type_of_target(y)
        if target_type in ['multiclass']:
            #class_count = Counter(y)
            #total_samples = sum(class_count.values())
            #relevance = self._is_imbalanced(class_count, total_samples)
            #if relevance:
            return True
        else:
            return False
            
    def suitable(self, y):
        return self.is_pertinent(y)
    
    # Specificity is calculated by summing the true negartives and false positives for each class, then using these totals to obtain an overall specificity"
    def compute(self, y, y_pred):
        cm = confusion_matrix(y, y_pred)
        total_tn = 0
        total_fp = 0
        for i in range(len(cm)):
            total_tn += np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            total_fp += np.sum(cm[:, i]) - cm[i, i]
        return total_tn / (total_tn + total_fp)