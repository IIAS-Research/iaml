"""
[METRIC] Parent of all others Metrics, implement the default behavior
"""
from typing import List
import pandas as pd
from .reference import Reference

class Metric:
    """
    [METRIC] Parent of all others Metrics, implement the default behavior
    """
    refs = []
    description_long = ""
    
    @classmethod
    def all_subclasses(cls) -> List['Metric']:
        """
        Return all metrics subclasses
        
        Returns
        -------
        List[Metric]
            List of all metrics
        """
        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            subclasses += subclass.all_subclasses()
        return subclasses

    @classmethod
    def get_refs(cls) -> List:
        """
        Get bibliography references
        
        Returns
        -------
        List[Reference]
            List of references for this metric
        """
        if hasattr(cls, 'refs'):
            return [Reference(ref, cls.__name__) for ref in cls.refs]
        return []
    
    def explain(self) -> str:
        """
        Describe metric

        Returns
        -------
        str
            Metric description
        """
        return self.description_long
    
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs):
        """
        Compute metric given y, y_pred. 
        Must be overwrote by children classes
        
        Parameters
        ----------
        y : pd.DataFrame
            Ground truth to compute the metric
        y_pred : pd.DataFrame
            Prediction to compute the metric
        
        Raises
        ------
        NotImplementedError
            Subclass must implement abstract method
        """
        raise NotImplementedError('Subclass must implement abstract method')
    
    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:  # pylint: disable=unused-argument
        """
        Does this metric is usable given X and y ?
        
        Parameters
        ----------
        X : pd.DataFrame
            The dataset we try to compute metrics on
        y : pd.DataFrame
            The dataset target we try to compute metrics on
        type_of_target : str
            The type of target we are trying to predict
        
        Returns
        -------
        bool
            Suitable ?
        """
        return False
    
    @property
    def needed_prediction(self) -> str:
        """
        Which kind of predict is needed by the metric

        Returns
        -------
        str
            method name
        """
        return "predict"

    @property
    def name(self) -> str:
        """
        Return the metric formatted name
        
        Returns
        -------
        str
            formatted name
        """
        return ' '.join(x.title() for x in str(self).split('_'))
