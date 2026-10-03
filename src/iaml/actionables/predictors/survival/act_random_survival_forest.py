"""[STEP] Random Survival Forest"""
import textwrap
from typing import Any
from sksurv.ensemble import RandomSurvivalForest

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step
from ._survival_forest import SurvivalForestMemoryMixin, survival_forest_constraints


@is_step('predictor', 'tabular', 'survival')
class ActRandomSurvivalForest(SurvivalForestMemoryMixin, Predictor):
    """[STEP] Random Survival Forest"""

    name: str = "RandomSurvivalForest"
    _flow_parameter_constraints = survival_forest_constraints(RandomSurvivalForest)
    _usage: str = "Use when you want a flexible tree-ensemble survival model; choose over ActCox when PH is doubtful, or consider ActExtraSurvivalTrees for more randomness. Applicable to tabular time-to-event data with censoring. Avoid when data are tiny or effects are well modeled by linear PH."
    _description: str = textwrap.dedent('''\
        RandomSurvivalForest is a survival analysis algorithm
        that uses an ensemble of decision trees to estimate the survival function
        over time. It is a non-parametric model that handles complex relationships
        and can model non-linear effects of covariates.''')
    _description_long: str = textwrap.dedent('''\
        RandomSurvivalForest is a flexible survival analysis
        algorithm that uses an ensemble of decision trees to predict the time
        until an event occurs. It extends the concept of random forests to survival
        data, handling complex interactions and non-linear relationships between
        input features (covariates). Unlike parametric models such as the Cox
        Proportional Hazards model, RandomSurvivalForest makes fewer assumptions
        about the underlying data, making it useful in cases where the assumptions
        of proportional hazards do not hold. It also efficiently manages censored
        data, where the event may not have occurred during the study period.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2008,
            'name': 'Random survival forests',
            'authors': [
                'H. Ishwaran',
                'U. B. Kogalur',
                'E. H. Blackstone',
                'M. S. Lauer'
            ],
            'doi': 'https://doi.org/10.1214/08-AOAS169',
            'publisher': 'The Annals of Applied Statistics, 2(3): page 841-860'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'n_estimators': {
                'description': 'Number of trees in the forest.',
                'default': 100,
                'range': [1, 1000],
                'passthrough': True
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 10,
                'range': [2, 20],
                'passthrough': True
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 5,
                'range': [1, 20],
                'passthrough': True
            },
            'max_depth': {
                'description': textwrap.dedent('''\
                    The maximum depth of the tree. If None, then nodes are
                    expanded until all leaves are pure.'''),
                'default': 12,
                'range': [1, None],
                'passthrough': True
            },
            'max_leaf_nodes': {
                'description': 'Maximum number of leaves per tree. Set None to remove this limit.',
                'default': 64,
                'range': [2, None],
                'passthrough': True
            },
            'random_state': {
                'description': 'Random seed (integer or None) for the estimator.',
                'default': None,
                'passthrough': True
            },
            'low_memory': {
                'description': (
                    "Use 'auto' to store risk scores only when every requested metric and "
                    'prediction requires only predict(). False preserves survival and hazard '
                    'curves; True requires a risk-only output contract.'
                ),
                'default': 'auto',
                'passthrough': True
            }
        }
        self.model: RandomSurvivalForest = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        return self._fit_survival_forest(RandomSurvivalForest, dataset)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
