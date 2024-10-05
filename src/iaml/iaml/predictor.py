"""
Last step of a pipeline -> can make prediction
"""
from abc import ABCMeta, abstractmethod
from typing import Any
import dataclasses
import pandas as pd
from sklearn.base import BaseEstimator
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

class Predictor(Actionable, BaseEstimator, metaclass=ABCMeta):
    """
    [STEP] Learn : Abstract learning step
    
    Also acts as an interface with traditional scikit-learn models
    for better integration with IAMLPipeline.
    """
    
    model: Model
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizable:bool = True
        self.model = None
    
    @runner
    def run(self, candidate:Candidate) -> Candidate:
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
        
        raise AttributeError("Unable to predict probabilities with this model")
    
    def __getattribute__(self, attr: str) -> bool:
        """Overload getattr to allow accurate hasattr on predict_proba

        Args:
            attr (str): Attribute to test

        Returns:
            bool: does attribute is implemented
        """
        if attr == 'predict_proba' \
            and not( \
                self.model and hasattr(self.model, 'predict_proba') \
            ):
            raise AttributeError("predict_proba not implemented in this model")
        
        return super().__getattribute__(attr)
    
    def predict(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        if self.model and hasattr(self.model, 'predict'):
            results = self.model.predict(X)
            if hasattr(self, 'label_encoder'):
                return self.label_encoder.inverse_transform(results)
            return results
        return None
    
    
    def predict_survival_function(self, X:pd.DataFrame) -> list[float]:
        """
        Apply prediction survival function model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[list[float]]: Predicted values
        """
        if self.model and hasattr(self.model, 'predict_survival_function'):
            return self.model.predict_survival_function(X)
        
        raise AttributeError("Unable to predict survival function with this model")
                
    @property
    def classes_(self) -> list:
        """Return classes of the target in fit data
        """
        return self.model.classes_
    
    def score(self, *args, **kwargs):
        """
        Mimic Scikitlearn API
        """
        return self.model.score(*args, **kwargs)
    
    def __name__(self) -> str:
        """Return the predictor formatted name
        
        Returns:
            str: formatted name
        """
        return ' '.join(x.title() for x in str(self).split('_'))
    