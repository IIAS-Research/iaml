"""
[METRIC] Parent of all others Metrics, implement the default behavior
"""
import pandas as pd

from .reference import Reference

class Metric:
    """
    [METRIC] Parent of all others Metrics, implement the default behavior
    """
    refs = []
    _description_long = ""
    
    @classmethod
    def all_subclasses(cls):
        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            subclasses += subclass.all_subclasses()
        return subclasses

    @classmethod
    def get_refs(cls):
        """
            Get bibliography references
        """
        if hasattr(cls, 'refs'):
            return [Reference(ref, cls.__name__) for ref in cls.refs]
        return []
    
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return self._description_long
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs):
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
    
    @property
    def needed_prediction(self) -> str:
        """Which kind of predict is needed by the metric

        Returns:
            string: method name
        """
        return "predict"

    @property
    def name(self) -> str:
        """Return the metric formatted name
        
        Returns:
            str: formatted name
        """
        return ' '.join(x.title() for x in str(self).split('_'))
