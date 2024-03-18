"""
Last step of a pipeline -> can make prediction
"""
import pandas as pd
from .actionable import Actionable
from .decorators.runner import runner
from .candidate import Candidate

class Predictor(Actionable):
    """
    Last step of a pipeline -> can make prediction
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    
    def model_parameters(self, default:bool=True):
        parameters = {}
        for key, value in self.configuration.items():
            if "model_parameter" in value:
                if value['model_parameter']:
                    parameters[key] = value['value']
            elif default:
                parameters[key] = value['value']
                
        return parameters
                
    @property
    def classes_(self):
        return self.model.classes_
