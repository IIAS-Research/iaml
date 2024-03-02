from ..step import *
from ..metric import Metric
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix

class SpecificityMetric(Metric):
    
    def __str__(self):
        return 'specificity'
    
    def explain(self):
        return 'The proportion of negative instances that are corectely classified as negative: tn / (tn + fp)'
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:
        return type_of_target in ['binary']

    def compute(self, y, y_pred):
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0