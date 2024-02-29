from ..step import *
from ..metric import Metric
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
    
    def suitable(self, y):
        target_type = type_of_target(y)
        if target_type in ['multilabel-indicator']:
            return True
        else:
            return False
    # Specificity is calculated for each label separately, then averaged to obtain an overall measure. 
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