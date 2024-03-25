"""
Last step of a pipeline -> can make prediction
"""
from abc import ABCMeta, abstractmethod
from typing import Any
import dataclasses
import pandas as pd
from .actionable import Actionable
from .decorators.runner import runner
from .candidate import Candidate

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

class Predictor(Actionable, metaclass=ABCMeta):
    """
    [STEP] Learn : Abstract learning step
    
    Also acts as an interface with traditional scikit-learn models
    for better integration with AutoPipeline.
    """
    
    model: Model
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizable:bool = True
        self.model = None
    
    @runner
    def run(self, candidate:Candidate, callback:callable=None) -> Candidate:
        """
        Run the step. In "Run" stage, predict does not "fit". Only add himself to pipeline

        Args:
            candidate (Candidate): Candidate informations 

        Returns:
            Candidate: transformed Candidate 
        """
        return candidate.add_to_pipeline(self)
    
    def predict_proba(self, X:pd.DataFrame) -> list[float]:
        """ 
        Apply prediction model on DataFrame with probability

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        if self.model and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        if self.model and hasattr(self.model, 'predict'):
            return self.model.predict(X)
        return None
                
    @property
    def classes_(self) -> list:
        """Return classes of the target in fit data
        """
        return self.model.classes_
