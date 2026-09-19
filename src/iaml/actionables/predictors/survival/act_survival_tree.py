"""[STEP] SurvivalTree"""
import textwrap
from typing import Any
from sksurv.tree import SurvivalTree

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival', 'baseline_predictor')
class ActSurvivalTree(Predictor):
    """[STEP] SurvivalTree"""

    name: str = "SurvivalTree"
    _description: str = textwrap.dedent('''\
        SurvivalTree is a decision tree algorithm tailored for survival analysis.
        It constructs a tree structure based on the log-rank test, where each split
        is designed to separate data by survival times. The method handles censored data
        and outputs risk scores, cumulative hazard functions, and survival functions
        based on the tree's terminal nodes.''')
    _description_long: str = textwrap.dedent('''\
        SurvivalTree builds a decision tree based on survival data, using the
        log-rank splitting rule to determine the best splits. It is a non-parametric model that
        is particularly suited for survival analysis with right-censored data. The model provides
        both cumulative hazard and survival functions at each terminal node, making it useful for
        clinical risk prediction and other applications where time-to-event outcomes are crucial.
        ''')
    _usage: str = "Use when you need an interpretable survival baseline; compare ActCox or ActExtraSurvivalTrees for linear or ensemble options. Applicable to tabular time-to-event data with right censoring. Avoid when higher accuracy is required or proportional-hazards structure is assumed."
    refs: list[dict[str, Any]] = [
        {
            'year': 1993,
            'name': 'Survival Trees by Goodness of Split',
            'authors': [
                'M. Leblanc',
                'J. Crowley'
            ],
            'doi': 'https://doi.org/10.1080/01621459.1993.10476296',
            'publisher': 'Journal of the American Statistical Association, 88(422), 457-467'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'splitter': {
                'description': textwrap.dedent('''\
                    The strategy used to split at each node. Supported: "best",
                    "random".'''),
                'default': 'best',
                'categorical': ['best', 'random'],
                'passthrough': True
            },
            'max_depth': {
                'description': 'The maximum depth of the tree.',
                'default': None,
                'range': [1, None],
                'passthrough': True
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 6,
                'range': [2, 20],
                'passthrough': True
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 3,
                'range': [1, 20],
                'passthrough': True
            },
            'min_weight_fraction_leaf': {
                'description': textwrap.dedent('''\
                    The minimum weighted fraction of the input samples required
                    to be at a leaf node.'''),
                'default': 0.0,
                'range': [0.0, 0.5],
                'passthrough': True
            },
            'max_features': {
                'description': textwrap.dedent('''\
                    The number of features to consider when looking for the
                    best split.'''),
                'default': None,
                'categorical': [None, 'sqrt', 'log2'],
                'passthrough': True
            },
            'random_state': {
                'description': 'Random seed (integer or None) for the estimator.',
                'default': None,
                'passthrough': True
            },
            'max_leaf_nodes': {
                'description': 'Grow a tree with a maximum number of leaf nodes.',
                'default': None,
                'range': [None, 1000],
                'passthrough': True
            },
            'low_memory': {
                'description': 'Reduce memory usage but disable some prediction functions.',
                'default': False,
                'categorical': [True, False],
                'passthrough': True
            }
        }
        self.model: SurvivalTree = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.model = SurvivalTree(
            **self.passthrough_parameters()
        )
        X, y = dataset.to_survival()
        self.model.fit(X, y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
