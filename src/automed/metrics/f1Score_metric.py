from ..step import *
from ..metric import Metric
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import pandas as pd
from sklearn.metrics import *

class F1ScoreMetric(Metric):
    
    def __str__(self):
        return 'f1_score'
    
    def explain(self):
        return 'Compute the F1 score, also known as balanced F-score or F-measure. The F1 score can be interpreted as a harmonic mean of the precision and recall'
    
            
    def suitable(self, y):
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass',  'multilabel-indicator']:
            return True
        else:
            return False
    
    def compute(self, y, y_pred):
        if type_of_target(y) == 'binary':
            return f1_score(y, y_pred, pos_label = y.unique()[0])
        elif type_of_target(y) == 'multiclass':
            return f1_score(y, y_pred, average ='weighted')
        elif type_of_target(y) == 'multilabel-indicator':
            return f1_score(y, y_pred, average ='samples')
        else:
            raise Exception('Metric not suitable') 