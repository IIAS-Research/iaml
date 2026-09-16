
"""[STEP]  Cox"""
import textwrap
from typing import Any
from sksurv.linear_model import CoxPHSurvivalAnalysis

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActCox(Predictor):
    """[STEP]  Cox"""

    name: str = "CoxPHSurvivalAnalysis"
    _description: str = textwrap.dedent('''\
        CoxPHSurvivalAnalysis is a survival analysis algorithm
        that estimates the effect of covariates on the likelihood of an event
        occurring over time, using the Cox proportional hazards model.''')
    _description_long: str = textwrap.dedent('''\
        CoxPHSurvivalAnalysis is a survival analysis method
        that models the relationship between multiple input features (covariates)
        and the time until a particular event happens. The algorithm is based on
        the Cox proportional hazards model, which assumes that the hazard or risk
        of an event is a product of a baseline hazard and a factor that depends on
        the covariates. This model is commonly used in medical research to study
        how factors such as age, treatment, or health conditions influence survival
        rates, or in engineering to predict equipment failure. Unlike many other
        models, it doesn't predict the exact time of the event but estimates the
        risk over time, handling cases where the event has not yet occurred (censored data).''')
    _usage: str = "Use when you need a Cox model; compare ActCoxnetSurvivalAnalysis for regularization or ActRandomSurvivalForest for nonlinearity. Applicable to tabular survival data with censoring and proportional hazards. Avoid when hazards are non-proportional or interactions dominate."
    refs: list[dict[str, Any]] = [
        {
            'year': 1972,
            'name': 'Regression models and life tables',
            'authors': [
                'D. R. Cox'
            ],
            'doi': 'https://doi.org/10.1111/j.2517-6161.1972.tb00899.x',
            'publisher': 'Journal of the Royal Statistical Society. Series B, 34: page 187-220'
        },
        {
            'year': 1974,
            'name': 'Covariance Analysis of Censored Survival Data',
            'authors': [
                'N. E. Breslow'
            ],
            'doi': 'https://doi.org/10.2307/2287816',
            'publisher': 'Biometrics, 30: page 89-99'
        },
        {
            'year': 1977,
            'name': 'The Efficiency of Cox’s Likelihood Function for Censored Data',
            'authors': [
                'B. Efron'
            ],
            'doi': 'https://doi.org/10.1007/978-0-387-75692-9_6',
            'publisher': 'Journal of the American Statistical Association, 72: page 557-565'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'alpha': {
                'description': textwrap.dedent('''\
                    Regularization strength. Higher values specify stronger
                    regularization. alpha=0 means no regularization.'''),
                'default': 1,
                'range': [0, 100],
                'passthrough': True
            },
            'ties': {
                'description': textwrap.dedent('''\
                    Method for handling tied event times in the data.
                    "breslow" is the most common method.'''),
                'default': 'breslow',
                'categorical': ['breslow', 'efron']
            },
            'n_iter': {
                'description': 'Maximum number of iterations for fitting the model.',
                'default': 100,
                'range': [1, 10000],
                'passthrough': True
            },
            'tol': {
                'description': textwrap.dedent('''\
                    Tolerance for stopping criteria. Determines the precision
                    of the solution.'''),
                'default': 1e-09,
                'range': [1e-12, 1e-03],
                'passthrough': True
            }
        }
        self.model: CoxPHSurvivalAnalysis = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = CoxPHSurvivalAnalysis(
            **self.passthrough_parameters()
            )
        X, y = dataset.to_survival()
        self.model.fit(X, y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
