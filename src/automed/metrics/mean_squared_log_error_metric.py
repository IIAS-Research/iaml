from ..step import *
from ..metric import Metric

from sklearn.metrics import mean_squared_log_error

class MeanSquaredLogErrorMetric(Metric):
    
    def __str__(self):
        return 'mean_squared_log_error'
    
    def explain(self):
        return 'Mean squared logarithmic error regression loss.'
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous' and not (dataset.Y < 0).any(axis=None)
    
    def compute(self, y, y_pred):
        try:
            return mean_squared_log_error(y, y_pred)
        except:
            return None