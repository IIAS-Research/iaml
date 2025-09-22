"""
[STEP] Gaussian Process Regressor
"""

import textwrap
from sklearn.gaussian_process import GaussianProcessRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActGaussianProcessRegressor(Predictor):
    """
    [STEP] Gaussian Process Regressor
    """
    name = "Gaussian Process Regressor"
    _description = textwrap.dedent('''\
        GaussianProcessRegressor is a machine learning algorithm
        that makes predictions for regression tasks using Gaussian processes.''')
    _description_long = textwrap.dedent('''\
        GaussianProcessRegressor is a powerful algorithm for regression tasks,
        especially when the relationship between the input features and the output variable is
        complex and non-linear, and when uncertainty estimates are important.''')
    refs = [
        {
            'year': 2006,
            'name': 'Gaussian Processes for Machine Learning',
            'authors': [
                'Carl Edward Rasmussen',
                'Christopher K. I. Williams'
            ],
            'doi': 'https://doi.org/10.7551/mitpress/3206.001.0001',
            'publisher': 'MIT Press 2006'
        }
    ]
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
    
    
    def suitable(self, dataset:Dataset) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return dataset.type_of_target == 'continuous'
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
