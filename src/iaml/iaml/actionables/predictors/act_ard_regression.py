"""
[STEP] Learn : ARD Regression
"""

from sklearn.linear_model import ARDRegression
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActARDRegression(Predictor):
    """
    [STEP] Learn : ARD Regression
    """
    name = "Learn : ARD Regression"
    def __init__(self):
        self.configuration:dict = {
            'alpha_1': {
                'description': 'Hyper-parameter : shape parameter for the Gamma \
                    distribution prior over the alpha parameter.',
                'default': 1e-06,
                'range': [1e-10, 0.001]
                },
            'alpha_2': {
                'description': 'Hyper-parameter : inverse scale parameter \
                    (rate parameter) for the Gamma distribution prior over \
                        the alpha parameter.',
                'default': 1e-06,
                'range': [1e-10, 0.001]
                },
            'lambda_1': {
                'description': 'Hyper-parameter : shape parameter for the Gamma \
                    distribution prior over the lambda parameter.',
                'default': 1e-10,
                'range': [0.001, 1e-06]
                },
            'lambda_2': {
                'description': 'Hyper-parameter : inverse scale parameter (rate parameter) \
                    for the Gamma distribution prior over the lambda parameter.',
                'default': 1e-10,
                'range': [0.001, 1e-06]
                },
            'threshold_lambda': {
                'description': 'Threshold for removing (pruning) weights with high \
                    precision from the computation.',
                'default': 10000.0,
                'range': [1000.0, 100000.0]
                },
            'tol': {
                'description': 'Stop the algorithm if w has converged.',
                'default': 0.001,
                'range': [1e-05, 0.1]
                }
            }
        self.model:ARDRegression = None
  
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit ARDRegression on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = ARDRegression(**self.passthrough_parameters())
        
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
