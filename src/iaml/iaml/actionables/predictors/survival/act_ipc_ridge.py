"""
[STEP]  IPCRidge
"""
import textwrap
from sksurv.linear_model import IPCRidge
import numpy as np

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'survival', 'baseline_predictor')
class ActIPCRidge(Predictor):
    """
    [STEP]  IPCRidge
    """
    name = "IPCRidge"
    description = textwrap.dedent('''\
        IPCRidge is a survival analysis algorithm 
        that estimates the effect of covariates on the likelihood of an event 
        occurring over time, using inverse probability censoring weights (IPCW) 
        with Ridge regularization.''')
    description_long = textwrap.dedent('''\
        IPCRidge is a survival analysis method that extends Ridge regression 
        to survival data by using inverse probability censoring weights (IPCW). 
        This approach accounts for censored data and helps improve the stability 
        of the model when dealing with high-dimensional datasets or multicollinearity 
        by adding a regularization term to the objective function. IPCRidge is useful 
        for linear survival models where regularization is needed to avoid overfitting.''')

    def __init__(self):
        self.configuration: dict = {
            'alpha': {
                'description': 'Regularization strength. Higher values specify \
                    stronger regularization. alpha=0 means no regularization.',
                'default': 1.0,
                'range': [0, 100],
                'passthrough': False
            },
            'max_iter': {
                'description': 'Maximum number of iterations for fitting the model.',
                'default': 1000,
                'range': [1, 10000],
                'passthrough': False
            },
            'tol': {
                'description': 'Tolerance for stopping criteria. \
                    Determines the precision of the solution.',
                'default': 1e-6,
                'range': [1e-12, 1e-03],
                'passthrough': False
            }
        }
        self.model: IPCRidge = None
        
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit IPCRidge on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = IPCRidge(
            alpha=self.configuration['alpha']['default'],
            max_iter=self.configuration['max_iter']['default'],
            tol=self.configuration['tol']['default']
        )
        
        X, y = dataset.to_survival()  # Assuming dataset.to_survival() prepares the data
        self.model.fit(X, y)
        
        return self
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize itself

        Return : continuous between 0 and 1
        """
        return 0.5  # neutral
