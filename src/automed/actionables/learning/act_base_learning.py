"""
[STEP] Learn :  Abstract learning step
"""

from abc import ABCMeta, abstractmethod
from typing import Any
import dataclasses
import pandas as pd
from ...actionable import Actionable
from ...output import Input


@dataclasses.dataclass
class Model(metaclass=ABCMeta):
    """
    Model type (use for typing purposes only).
    """

    @abstractmethod
    def predict(self, X, *args, **kw) -> Any:
        """
        Any predict method implemented by most ML frameworks.
        """


class ActBaseLearning(Actionable, metaclass=ABCMeta):
    """
    [STEP] Learn : Abstract learning step

    Also acts as an interface with traditional scikit-learn models
    for better integration with AutoPipeline.
    """

    model: Model

    def predict(self, X: pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)

    def priorize(self, input_data: Input = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral

    @abstractmethod
    def suitable(self, input_data: Input) -> bool:
        """
        Checks whether this step is suitable for this input.

        Args:
            input_data (Input): Input

        Returns:
            bool: Whether the step is suitable.
        """
