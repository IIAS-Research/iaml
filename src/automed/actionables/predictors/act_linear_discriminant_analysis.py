"""
[STEP] Learn : Linear Discriminant Analysis
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActLinearDiscriminantAnalysis(Predictor):
    """
    [STEP] Learn : Linear Discriminant Analysis
    """
    name = "Learn : Linear Discriminant Analysis"
    def __init__(self):
        self.configuration:dict = {
            'tol': {
                'description': 'Absolute threshold for a singular value of X to be considered \
                    significant, used to estimate the rank of X. ',
                'default': 0.0001,
                'range': [1e-05, 0.1]
                }
            }
        self.model:LinearDiscriminantAnalysis = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Linear Discriminant Analysis on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = LinearDiscriminantAnalysis(**self.passthrough_parameters())
        
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
