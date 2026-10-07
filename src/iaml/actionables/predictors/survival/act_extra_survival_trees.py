"""[STEP] Extra Survival Trees"""
import textwrap
from typing import Any
from sksurv.ensemble import ExtraSurvivalTrees

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step
from ._survival_forest import SurvivalForestMemoryMixin, survival_forest_constraints


@is_step('predictor', 'tabular', 'survival')
class ActExtraSurvivalTrees(SurvivalForestMemoryMixin, Predictor):
    """[STEP] Extra Survival Trees"""
    name: str = "ExtraSurvivalTrees"
    _flow_parameter_constraints = survival_forest_constraints(ExtraSurvivalTrees)
    _usage: str = "Use when you want a randomized tree ensemble for survival, as an alternative to ActRandomSurvivalForest. Applicable to tabular censored survival data with nonlinear feature effects. Avoid when you need proportional-hazards interpretability like ActCox or data is very small."
    _description: str = textwrap.dedent('''\
        ExtraSurvivalTrees is an ensemble learning method for survival
        analysis based on extremely randomized trees. It fits multiple decision trees
        to the data, where each tree is built from a random subset of features and
        splits are selected randomly. This method provides more variance reduction
        and robustness, especially useful when dealing with high-dimensional or
        sparse data.''')
    _description_long: str = textwrap.dedent('''\
        ExtraSurvivalTrees is a variant of ensemble learning for survival
        analysis that uses extremely randomized trees. In this approach, multiple trees
        are grown by selecting random subsets of features and splitting points.
        Compared to other tree-based methods, this randomness helps reduce overfitting
        and increases model robustness. The method is particularly useful for survival
        datasets that contain complex, non-linear relationships between features.
        ExtraSurvivalTrees handles censored data and can provide interpretable models
        for survival time predictions.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Extremely Randomized Trees',
            'authors': [
                'P. Geurts',
                'D. Ernst',
                'L. Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning, 63(1), 3-42'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'n_estimators': {
                'description': 'The number of trees in the forest.',
                'default': 100,
                'range': [1, 1000],
                'passthrough': True
            },
            'max_depth': {
                'description': 'The maximum depth of the trees. Set None to remove this limit.',
                'default': 12,
                'range': [1, None],
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
            'max_features': {
                'description': textwrap.dedent('''\
                    The number of features to consider when looking for the
                    best split.'''),
                'default': "sqrt",
                'categorical': ["sqrt", "log2", None],
                'passthrough': True
            },
            'random_state': {
                'description': 'Random seed (integer or None) for the estimator.',
                'default': None,
                'passthrough': True
            },
            'max_leaf_nodes': {
                'description': 'Maximum number of leaves per tree. Set None to remove this limit.',
                'default': 64,
                'range': [2, None],
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
        self.model: ExtraSurvivalTrees = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        return self._fit_survival_forest(ExtraSurvivalTrees, dataset)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
