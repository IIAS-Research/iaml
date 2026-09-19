"""[STEP] Gradient Boosting Survival Analysis"""
import textwrap
from typing import Any
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival', 'minimal_predictor')
class ActGradientBoostingSurvivalAnalysis(Predictor):
    """[STEP] Gradient Boosting Survival Analysis"""

    name: str = "GradientBoostingSurvivalAnalysis"
    _usage: str = "Use when you need non-linear survival modeling and ActCox underfits. Applicable to tabular time-to-event data with right-censoring. Avoid when you need simpler baselines or strong ensembles like ActRandomSurvivalForest."
    _description: str = textwrap.dedent('''\
        GradientBoostingSurvivalAnalysis is a survival analysis algorithm
        that uses gradient boosting to model the risk of an event over time.
        It fits an ensemble of regression trees to capture non-linear effects
        and interactions in censored survival data.''')
    _description_long: str = textwrap.dedent('''\
        GradientBoostingSurvivalAnalysis extends gradient boosting to
        time-to-event data by optimizing a survival-specific loss function.
        The model builds an ensemble of shallow regression trees, each correcting
        the errors of the previous ones, resulting in a flexible estimator for
        complex covariate effects. It can handle right-censored observations and
        is useful when proportional hazards assumptions are too restrictive.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2010,
            'name': 'Gradient boosting for survival analysis',
            'authors': [
                'Chen, Yifei',
                'Jia, Zhenyu',
                'Mercola, Dan',
                'Xie, Xiaohui'
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
                'passthrough': True
            },
            'learning_rate': {
                'description': 'Learning rate shrinks the contribution of each tree by this value.',
                'default': 0.1,
                'range': [0.01, 1.0],
                'passthrough': True
            },
            'max_depth': {
                'description': 'The maximum depth of the individual trees.',
                'default': 3,
                'range': [1, 20],
                'passthrough': True
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 2,
                'range': [2, 20],
                'passthrough': True
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 20],
                'passthrough': True
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
