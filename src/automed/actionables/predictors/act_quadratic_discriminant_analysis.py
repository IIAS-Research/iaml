"""
[STEP] Learn : Quadratic Discriminant Analysis
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActQuadraticDiscriminantAnalysis(Predictor):
    """
    [STEP] Learn : Quadratic Discriminant Analysis
    """
    name = "Learn : Quadratic Discriminant Analysis"
    def __init__(self):
        self.configuration:dict = {
            'reg_param': {
                'description': 'Regularizes the per-class covariance estimates by transforming S2',
                'default': 0.0001,
                'range': [0.0001, 1.0]
                }
            }
        self.model:QuadraticDiscriminantAnalysis = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Quadratic Discriminant Analysis on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = QuadraticDiscriminantAnalysis(**self.model_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, candidate:Candidate) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
