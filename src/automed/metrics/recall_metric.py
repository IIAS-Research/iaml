from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import *

class RecallMetric(Metric):
    
    def __str__(self):
        return 'recall'
    
    def explain(self):
        return 'The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.'
    
    def suitable(self, dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass',  'multilabel-indicator']
        
    def compute(self, y, y_pred):
        if type_of_target(y) == 'binary':
            return precision_score(y, y_pred, pos_label=y.unique()[0]) # TODO Find something less arbitrary
        elif type_of_target(y) == 'multiclass':
            return precision_score(y, y_pred, average = 'weighted') 
        elif type_of_target(y) == 'multilabel-indicator':
            return precision_score(y, y_pred, average= 'samples')
        else:
            raise Exception('Metric not suitable') 