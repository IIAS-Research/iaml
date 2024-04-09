"""
[METRIC] Parent of all others Metrics, implement the default behavior
"""
import pandas as pd
class Metric:
    """
    [METRIC] Parent of all others Metrics, implement the default behavior
    """
    def explain(self) -> str:
        """
        Return str explanation of the metric
        """
        return "Here is an explanation of how this metrics work"
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame):
        """
        Compute metric given y, y_pred. 
        Must be overwrote by children classes
        """
        raise NotImplementedError('Subclass must implement abstract method')
    
    def suitable(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this metric is usable given X and y ?
        """
        return False
    
    def need_proba(self):
        return False
