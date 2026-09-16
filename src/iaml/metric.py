"""[METRIC] Parent of all others Metrics, implement the default behavior"""
from typing import Any
import pandas as pd
from .reference import Reference


class Metric:
    """[METRIC] Parent of all others Metrics, implement the default behavior"""

    greater_is_better: bool = True
    """Whether larger values are better. Set to False for error metrics."""

    name: str = ""
    """Name of the metric"""

    _description: str = ""
    """Short description of the metric"""

    _description_long: str = ""
    """Long description of the metric"""

    refs: list[dict[str, Any]] = []
    """List of references for this metric."""

    @classmethod
    def all_subclasses(cls) -> list['Metric']:
        """Return all metrics subclasses
        
        :return: List of all metrics.
        """
        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            subclasses += subclass.all_subclasses()
        return subclasses

    @classmethod
    def get_refs(cls) -> list[Reference]:
        """Get bibliography references
        
        :return: List of references for this metric.
        """
        if hasattr(cls, 'refs'):
            return [Reference(ref, cls.__name__) for ref in cls.refs]
        return []

    @property
    def description(self) -> str:
        """Formats the description of a step with its configuration.

        :return: Formatted description.
        """
        return self._description.replace('\n', '')

    @property
    def description_long(self) -> str:
        """Formats the longer description of a step with its configuration.
        
        :return: Formatted description.
        """
        return self._description_long.replace('\n', '')

    def __str__(self) -> str:
        """Metric name
        
        :return: Metric name.
        """
        return 'base_metric'

    def explain(self) -> str:
        """Describe metric

        :return:    Metric description
        """
        return self.description_long

    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
        """Compute metric given y, y_pred. 
        Must be overridden by children classes
        
        :param pd.DataFrame y: Ground truth to compute the metric.
        :param pd.DataFrame y_pred: Prediction to compute the metric.
        :param dict, optional \\**kwargs: Additional parameters
        :return: Computed value
        """
        raise NotImplementedError('Subclass must implement abstract method')

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:  # pylint: disable=unused-argument
        """Is this metric usable given X and y ?
        
        :param pd.DataFrame X: The dataset we try to compute metrics on.
        :param pd.DataFrame y: The dataset target we try to compute metrics on
        :param str type_of_target: The type of target we are trying to predict
        :return: Suitable ?
        """
        return False

    @property
    def needed_prediction(self) -> str:
        """Which kind of predict is needed by the metric

        :return: Method name.
        """
        return "predict"

    @property
    def name(self) -> str:
        """Return the metric formatted name
        
        :return: Formatted name.
        """
        return ' '.join(x.title() for x in str(self).split('_'))
