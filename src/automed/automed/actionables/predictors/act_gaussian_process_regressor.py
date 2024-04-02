"""
[STEP] Learn : Gaussian Process Regressor
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActGaussianProcessRegressor(Predictor):
    """
    [STEP] Learn : Gaussian Process Regressor
    """
    name = "Learn : Gaussian Process Regressor"
    def __init__(self):
        self.configuration:dict = {
            'alpha': {
                'description': 'Value added to the diagonal of the kernel matrix during fitting.',
                'default': 1e-14,
                'range': [1e-08, 1.0]
                }
            }
        self.model:GaussianProcessRegressor = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit GaussianProcessRegressor on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = GaussianProcessRegressor(**self.passthrough_parameters())
        
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
        return candidate.dataset.type_of_target == 'continuous'
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
