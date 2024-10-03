
"""
[STEP] Learn :  Cox
"""
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'survival')
class ActCox(Predictor):
    """
    [STEP] Learn :  Cox
    """
    name = "Learn : CoxPHSurvivalAnalysis"
    description = '''CoxPHSurvivalAnalysis is a survival analysis algorithm 
        that estimates the effect of covariates on the likelihood of an event 
        occurring over time, using the Cox proportional hazards model.'''
    description_long = '''CoxPHSurvivalAnalysis is a survival analysis method 
        that models the relationship between multiple input features (covariates) 
        and the time until a particular event happens. The algorithm is based on 
        the Cox proportional hazards model, which assumes that the hazard or risk 
        of an event is a product of a baseline hazard and a factor that depends on 
        the covariates. This model is commonly used in medical research to study 
        how factors such as age, treatment, or health conditions influence survival 
        rates, or in engineering to predict equipment failure. Unlike many other 
        models, it doesn't predict the exact time of the event but estimates the 
        risk over time, handling cases where the event has not yet occurred (censored data).'''

    refs = [
        {
            'year': 1972,
            'name': 'Regression models and life tables (with discussion)',
            'authors': [
                'D. R. Cox'
            ],
            'doi': '',
            'publisher': 'Journal of the Royal Statistical Society. Series B, 34: page 187-220'
        },
        {
            'year': 1974,
            'name': 'Covariance Analysis of Censored Survival Data',
            'authors': [
                'N. E. Breslow'
            ],
            'doi': '',
            'publisher': 'Biometrics, 30: page 89-99'
        },
        {
            'year': 1977,
            'name': 'The Efficiency of Cox’s Likelihood Function for Censored Data',
            'authors': [
                'B. Efron'
            ],
            'doi': '',
            'publisher': 'Journal of the American Statistical Association, 72: page 557-565'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'alpha': {
                'description': 'Regularization strength. Higher values specify \
                    stronger regularization. alpha=0 means no regularization.',
                'default': 0,
                'range': [0, 100],
                'passthrough': False
            },
            'ties': {
                'description': 'Method for handling tied event times in the data. \
                    "breslow" is the most common method.',
                'default': 'breslow',
                'categorical': ['breslow', 'efron']
            },
            'n_iter': {
                'description': 'Maximum number of iterations for fitting the model.',
                'default': 100,
                'range': [1, 10000],
                'passthrough': False
            },
            'tol': {
                'description': 'Tolerance for stopping criteria. \
                    Determines the precision of the solution.',
                'default': 1e-09,
                'range': [1e-12, 1e-03],
                'passthrough': False
            }
        }
        self.model:CoxPHSurvivalAnalysis = None
        
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Knn classifier on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = CoxPHSurvivalAnalysis(
            **self.passthrough_parameters()
            )
        
        y = np.array(dataset.y, dtype=[('event', 'bool'), ('time', 'float')])
        self.model.fit(dataset.X, y)
        
        return self
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
