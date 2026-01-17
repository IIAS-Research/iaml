"""[STEP]  Random Forest"""
import textwrap
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActRandomForest(Predictor):
    """[STEP]  Random Forest"""

    name: str = "Random Forest"
    _usage: str = "Use when you need a solid tabular classifier baseline and prefer it over ActExtraTreesClassifier for stability. Applicable to binary, multiclass, or multilabel tabular data with tree-ready features. Avoid when data is extremely sparse text or you need native categorical handling."
    _description: str = textwrap.dedent('''\
        RandomForestClassifier is a machine learning algorithm that models
        the relationship between input features and a categorical output variable using
        a collection of decision trees.''')
    _description_long: str = textwrap.dedent('''\
        RandomForestClassifier is a type of ensemble learning algorithm that
        models the relationship between input features and a categorical output variable using
        a collection of decision trees. It works by building multiple decision trees on random
        subsets of the input features and data, and then using a majority vote to make
        the final prediction.''')
    refs: tuple[dict[str, Any]] = [
        {
            'year': 2001,
            'name': 'Random Forests',
            'authors': ['Leo Breiman'],
            'doi': 'https://doi.org/10.1023/A:1010933404324',
            'publisher': 'Machine Learning Vol.45 page 5--32'
        }
    ]

    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node',
                'default': 2,
                'range': [2, 20]
            },
            'bootstrap': {
                'description': textwrap.dedent('''\
                    Whether bootstrap samples are used when building trees. If
                    False, the whole dataset is used to build each tree.'''),
                'default': False
            },
            'criterion': {
                'description': 'The function to measure the quality of a split.',
                'default': "gini",
                'categorical': ['gini', 'entropy', 'log_loss']
            }
        }
        self.model: RandomForestClassifier = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = RandomForestClassifier(**self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
