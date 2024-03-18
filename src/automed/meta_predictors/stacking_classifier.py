"""
Ensemble based predict method
"""
import pandas as pd
from sklearn.ensemble import StackingClassifier as SKStackingClassifier
from ..meta_predictor import MetaPredictor
from ..candidate import Candidate
from ..auto_pipeline import AutoPipeline
from ..dataset import Dataset

class StackingClassifier(MetaPredictor):
    """
    Ensemble based predict method
    """
    def __init__(self, candidates: list[Candidate]):
        super().__init__(candidates)
        self.model = SKStackingClassifier(estimators=self.estimators)
    
    def fit(self, dataset:Dataset):
        self.model.fit(dataset.X, dataset.y)
        return self        
        
    @classmethod
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this meta predictor is usable given type of target ?
        """
        return type_of_target in ['binary', 'multiclass']
