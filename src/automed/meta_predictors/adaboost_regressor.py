"""
Ensemble based predict method
"""
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import AdaBoostRegressor as SKAdaBoostRegressor
from ..meta_predictor import MetaPredictor
from ..candidate import Candidate
from ..auto_pipeline import AutoPipeline
from ..dataset import Dataset

# class AdaBoostRegressor(MetaPredictor):
class AdaBoostRegressor():
    """
    Ensemble based predict method
    """
    def __init__(self, candidates: list[Candidate]):
        candidate = deepcopy(candidates[0])
        super().__init__([candidate])
        
        predictor = candidate.pipeline.predictor[1]
        
        self.model = SKAdaBoostRegressor(estimator=predictor)
        
    def to_candidate(self) -> Candidate:
        '''
        Create a candidate for meta predictor
        '''
        transformers_steps = self.estimators[0][1].training_steps
        candidate = Candidate(
            auto_pipeline=AutoPipeline(
                [*transformers_steps, ("AdaBoostRegressor", self)],
                estimator_type=self.estimators[0][1].estimator_type
                ),
            metrics=self.metrics
            )
        candidate.is_meta = True
        
        return candidate 
    
    def fit(self, dataset:Dataset):
        self.model.fit(dataset.X, dataset.y)
        return self
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this meta predictor is usable given type of target ?
        """
        return type_of_target == 'continuous'
