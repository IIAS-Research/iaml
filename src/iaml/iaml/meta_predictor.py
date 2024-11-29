"""
Ensemble based predict method
"""
from typing import List
from .candidate import Candidate
from .iaml_pipeline import IAMLPipeline
from .predictor import Predictor

class MetaPredictor(Predictor):
    """
    Ensemble based predict method
    """
    def __init__(self, candidates: List['Candidate']):
        """
        Initialize MetaPredictor
        
        Parameters
        ----------
        candidates : List[Candidate]
            List of candicate that'll be added to our MetaPredictor
        """
        super().__init__()
        self.configuration = {}
        self.metrics: list = candidates[0].metrics
        self.estimator_type: str = candidates[0].pipeline._estimator_type
        self.model = None
        self.estimators: list = [(f'{idx} - {candidate.pipeline.predictor[0]}', candidate.pipeline) \
            for idx, candidate in enumerate(candidates)]
        
    def to_candidate(self) -> Candidate:
        '''
        Create a candidate for meta predictor
        
        Returns
        -------
        Candidate
            The newly created candidate
        '''
        candidate = Candidate(
            iaml_pipeline=IAMLPipeline(
                [("Voting classifier", self)],
                estimator_type=self.estimator_type
                ),
            metrics=self.metrics
            )
        candidate.is_meta = True
        
        return candidate 
    
    def suitable(self, type_of_target: str) -> bool:  # pylint: disable=unused-argument, arguments-renamed
        """
        Does this meta predictor is usable given type of target ?
        
        Parameters
        ----------
        type_of_target : str
            The dataset type of target
        
        Returns
        -------
        bool
            Suitable ?
        """
        return False
