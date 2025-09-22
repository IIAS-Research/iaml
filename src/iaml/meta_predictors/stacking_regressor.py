"""Ensemble based predict method"""
from typing import TYPE_CHECKING
from sklearn.ensemble import StackingRegressor as SKStackingRegressor
from ..meta_predictor import MetaPredictor
from ..dataset import Dataset

if TYPE_CHECKING:
    from ..candidate import Candidate

class StackingRegressor(MetaPredictor):
    """Ensemble based predict method"""

    def __init__(self, candidates: list['Candidate']):
        super().__init__(candidates)
        self.model = SKStackingRegressor(estimators=self.estimators)

    def fit(self, dataset: Dataset):
        self.model.fit(dataset.X, dataset.y)
        return self

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:  # pylint: disable=arguments-differ
        return type_of_target == 'continuous'
