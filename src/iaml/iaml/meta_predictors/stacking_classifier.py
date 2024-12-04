"""Ensemble based predict method"""
from sklearn.ensemble import StackingClassifier as SKStackingClassifier
from ..meta_predictor import MetaPredictor
from ..candidate import Candidate
from ..dataset import Dataset

class StackingClassifier(MetaPredictor):
    """Ensemble based predict method"""
    def __init__(self, candidates: list[Candidate]):
        super().__init__(candidates)
        self.model = SKStackingClassifier(estimators=self.estimators)

    def fit(self, dataset: Dataset):
        self.model.fit(dataset.X, dataset.y)
        return self

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:  # pylint: disable=arguments-differ
        return type_of_target in ['binary', 'multiclass']
