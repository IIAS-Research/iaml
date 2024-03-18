"""
Ensemble based predict method
"""
from sklearn.ensemble import StackingRegressor as SKStackingRegressor
from ..meta_predictor import MetaPredictor
from ..candidate import Candidate
from ..dataset import Dataset

class StackingRegressor(MetaPredictor):
    """
    Ensemble based predict method
    """
    def __init__(self, candidates: list[Candidate]):
        super().__init__(candidates)
        self.model = SKStackingRegressor(estimators=self.estimators)
    
    def fit(self, dataset:Dataset):
        self.model.fit(dataset.X, dataset.y)
        return self        
        
    @classmethod
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this meta predictor is usable given type of target ?
        """
        return type_of_target == 'continuous'
