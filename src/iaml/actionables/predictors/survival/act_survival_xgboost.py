"""[STEP] Gradient Boosting Survival Analysis"""
import textwrap
from typing import Any
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActGradientBoostingSurvivalAnalysis(Predictor):
    """[STEP] Gradient Boosting Survival Analysis"""

    name: str = "GradientBoostingSurvivalAnalysis"
    _description: str = textwrap.dedent('''\
        GradientBoostingSurvivalAnalysis is a survival analysis algorithm
        that uses gradient boosting to estimate the survival function over time.
        It is an ensemble-based model that minimizes a differentiable loss function
        to predict the time until an event occurs.''')
    _description_long: str = textwrap.dedent('''\
        GradientBoostingSurvivalAnalysis is a flexible survival
        analysis algorithm that uses gradient boosting to predict the time until
        an event occurs. It extends the concept of boosting to survival data,
        handling complex interactions and non-linear relationships between input
        features (covariates). Unlike parametric models such as the Cox Proportional
        Hazards model, GradientBoostingSurvivalAnalysis makes fewer assumptions
        about the underlying data, making it useful in cases where the assumptions
        of proportional hazards do not hold. It also efficiently manages censored
        data, where the event may not have occurred during the study period.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2010,
            'name': 'Gradient boosting for survival analysis',
            'authors': [
                "Chen, Yifei",
                "Jia, Zhenyu",
                "Mercola, Dan",
                "Xie, Xiaohui"
            ],
            'doi': 'https://doi.org/10.1155/2013/873595',
            'publisher': 'Advances in Data Analysis, Data Handling and Business Intelligence, \
                pages 239-248'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'n_estimators': {
                'description': 'Number of boosting stages to be run.',
                'default': 100,
                'range': [1, 1000],
                'passthrough': False
            },
            'learning_rate': {
                'description': 'Learning rate shrinks the contribution of each tree by this value.',
                'default': 0.1,
                'range': [0.01, 1.0],
                'passthrough': False
            },
            'max_depth': {
                'description': 'The maximum depth of the individual trees.',
                'default': 3,
                'range': [1, 20],
                'passthrough': False
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 2,
                'range': [2, 20],
                'passthrough': False
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 20],
                'passthrough': False
            }
        }
        self.model: GradientBoostingSurvivalAnalysis = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.model = GradientBoostingSurvivalAnalysis(
            **self.passthrough_parameters()
        )
        X, y = dataset.to_survival()
        self.model.fit(X, y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
