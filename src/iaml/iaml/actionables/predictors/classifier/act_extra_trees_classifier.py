"""[STEP] Extra Trees Classifier"""
import textwrap
from typing import Any
from sklearn.ensemble import ExtraTreesClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActExtraTreesClassifier(Predictor):
    """[STEP] Extra Trees Classifier"""

    name: str = "Extra Trees Classifier"
    _description: str = textwrap.dedent('''\
        ExtraTreesClassifier is a machine learning algorithm that makes
        predictions by combining the outputs of multiple decision trees.''')
    _description_long: str = textwrap.dedent('''\
        ExtraTreesClassifier is a type of ensemble learning algorithm that
        belongs to the family of decision tree-based models. It works by building multiple
        decision trees, where each tree is trained on a random subset of the input feature
        and a random subset of the training data. At prediction time, the algorithm aggregates
        the outputs of all the decision trees to make a final prediction.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Extremely randomized trees',
            'authors': [
                'Pierre Geurts',
                'Damien Ernst',
                'Louis Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol. 63 page 3--42'
        }
    ]
    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
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
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 'sqrt',
                'categorical': ['sqrt', 'log2']
            },
            'criterion': {
                'description': 'The function to measure the quality of a split.',
                'default': "gini",
                'categorical': ['gini', 'entropy']
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model: ExtraTreesClassifier = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = ExtraTreesClassifier(**self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
