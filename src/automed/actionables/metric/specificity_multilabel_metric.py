from ...step import *
from ...metric import Metric 
from sklearn.utils.multiclass import type_of_target 
from collections import Counter 
import pandas as pd
from sklearn.metrics import *
import numpy as np

class SpecificityMultilabelMetric(Metric):
    
    def explain(self):
        return 'Compute specificity for each label separately, then averaged to obtain an overall measure.'
    
    def _str__(self):
        return 'specificity_multilabel'
    
    def _is_imbalanced(self, class_counts, total_samples):
        ideal_count = total_samples / len(class_counts)
        threshold = 0.20 * ideal_count
        return any(abs(count - ideal_count) > threshold for count in class_counts.values())
    
    def is_pertinent(self, y):
        target_type = type_of_target(y)
        if target_type in ['multilabel-indicator']:
            #class_count = Counter(y)
            #total_samples = sum(class_count.values())
            #relevence = self._is_imbalanced(class_count, total_samples)
            #if relevence :
            return True
            #else: 
            #    return False
            
    def suitable(self, y):
        return self.is_pertinent(y)
    
    def compute(self, y, y_pred):
        # Generates a series of confusion matrices,  one for each label
        mcm = multilabel_confusion_matrix(y, y_pred)
        specificity_per_label = []
        for i in range(mcm.shape[0]):
                tn, fp, fn, tp = mcm[i].ravel()
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
                specificity_per_label.append(specificity)
                
        mean_specificity = np.mean(specificity_per_label)
        return mean_specificity