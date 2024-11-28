"""
Last step of a pipeline -> can make prediction
"""
from abc import ABCMeta, abstractmethod
from typing import Any, List
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
    def predict(self, X: Any, *args, **kw) -> Any:
        """
        Any predict method implemented by most ML frameworks.
        
        Parameters
        ----------
        X: Any
            The dataset to predict 
        
        Returns
        -------
        Any
            Prediction output
        """

class Predictor(Actionable, BaseEstimator, metaclass=ABCMeta):
    """
    [STEP] Abstract learning step
    
    Also acts as an interface with traditional scikit-learn models
    for better integration with IAMLPipeline.
    """
    
    model: Model
    
    def __init__(self):
        super().__init__()
        self.optimizable:bool = True
        self.model = None
    
    @runner
    def run(self, candidate: Candidate) -> Candidate:
        """
        Run the step. In "Run" stage, predict does not "fit". Only add himself to pipeline

        Parameters
        ----------
        candidate : Candidate
            Candidate informations 

        Returns
        -------
        Candidate
            transformed Candidate 
        """
        return candidate.add_to_pipeline(self)
    
    def predict_proba(self, X: pd.DataFrame) -> List[float]:
        """ 
        Apply prediction model on DataFrame with probability

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame use to predict

        Returns
        -------
        List[float]
            Predicted values
        """
        if self.model and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        
        raise AttributeError("Unable to predict probabilities with this model")
    
    def __getattribute__(self, attr: str) -> bool:
        """Overload getattr to allow accurate hasattr on predict_proba

        Parameters
        ----------
        attr : str
            Attribute to test

        Returns
        -------
        bool
            does attribute is implemented
        """
        if attr == 'predict_proba' \
            and not( \
                self.model and hasattr(self.model, 'predict_proba') \
            ):
            raise AttributeError("predict_proba not implemented in this model")
        
        return super().__getattribute__(attr)
    
    def predict(self, X: pd.DataFrame) -> List[float]:
        """
        Apply prediction model on DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame use to predict

        Returns
        -------
        List[float]
            Predicted values
        """
        if self.model and hasattr(self.model, 'predict'):
            results = self.model.predict(X)
            if hasattr(self, 'label_encoder'):
                return self.label_encoder.inverse_transform(results)
            return results
        return None
    
    
    def predict_survival_function(self, X: pd.DataFrame) -> List[List[float]]:
        """
        Apply prediction survival function model on DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame use to predict

        Returns
        -------
        List[List[float]]
            Predicted values
        """
        if self.model and hasattr(self.model, 'predict_survival_function'):
            return self.model.predict_survival_function(X)
        
        raise AttributeError("Unable to predict survival function with this model")
    
    def predict_cumulative_hazard_function(self, X: pd.DataFrame) -> List[List[float]]:
        """
        Apply prediction survival function model on DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame use to predict

        Returns
        -------
        List[List[float]]: Predicted values
        """
        if self.model and hasattr(self.model, 'predict_cumulative_hazard_function'):
            return self.model.predict_cumulative_hazard_function(X)
        
        raise AttributeError("Unable to predict cumulative hazard function with this model")
                
    @property
    def classes_(self) -> List:
        """Return classes of the target in fit data
        """
        return self.model.classes_
    
    def score(self, *args, **kwargs) -> Any:
        """
        Mimic Scikitlearn API
        
        Returns
        -------
        Any
            The model score
        """
        return self.model.score(*args, **kwargs)
    
    def get_params(self, *args, **kwargs) -> Any | None:
        
        """
        Mimic Scikitlearn API
        
        Returns
        -------
        Any | None
            Model parameters or None if no parameters
        """
        if self.model and hasattr(self.model, 'get_params'):
            return self.model.get_params(*args, **kwargs)
        return None
    
    def __name__(self) -> str:
        """Return the predictor formatted name
        
        Returns:
            str: formatted name
        """
        return ' '.join(x.title() for x in str(self).split('_'))
