"""[STEP] Extra Survival Trees"""
import textwrap
from typing import Any
from sksurv.ensemble import ExtraSurvivalTrees

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActExtraSurvivalTrees(Predictor):
    """[STEP] Extra Survival Trees"""
    name: str = "ExtraSurvivalTrees"
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
                'passthrough': False
            },
            'max_depth': {
                'description': 'The maximum depth of the trees.',
                'default': None,
                'range': [1, None],
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
            },
            'max_features': {
                'description': textwrap.dedent('''\
                    The number of features to consider when looking for the
                    best split.'''),
                'default': "sqrt",
                'options': ["auto", "sqrt", "log2", None],
                'passthrough': False
            },
            'random_state': {
                'description': 'Controls the randomness of the estimator.',
                'default': None,
                'options': [None, 'int'],
                'passthrough': False
            }
        }
        self.model: ExtraSurvivalTrees = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.model = ExtraSurvivalTrees(
            **self.passthrough_parameters()
        )
        X, y = dataset.to_survival()
        self.model.fit(X, y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
