"""[STEP]  Decision Tree Classifier"""
import textwrap
from typing import Any
from sklearn.tree import DecisionTreeClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActDecisionTreeClassifier(Predictor):
    """[STEP]  Decision Tree Classifier"""

    name: str = "Decision Tree Classifier"
    _description: str = textwrap.dedent('''\
        DecisionTreeClassifier is a tree-based algorithm that learns
        if-then rules to predict a categorical target.''')
    _description_long: str = textwrap.dedent('''\
        DecisionTreeClassifier builds a tree by recursively splitting the data
        on feature thresholds to reduce impurity. The resulting tree can be
        inspected for interpretability.''')
    _usage: str = "Use when you want a fast, interpretable baseline; compare ActExtraTreesClassifier or ActBaggingClassifier. Applicable to tabular classification with numeric features and binary or multiclass targets. Avoid when you need top accuracy or stability on noisy data."
    refs: list[dict[str, Any]] = [
        {
            'year': 1984,
            'name': 'Classification and Regression Trees',
            'authors': [
                'Leo Breiman',
                'Jerome Friedman',
                'Richard Olshen',
                'Charles Stone'
            ],
            'publisher': 'Wadsworth'
        }
    ]

    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Maximum depth of the tree.',
                'default': 5,
                'range': [1, 50]
            },
            'min_samples_leaf': {
                'description': 'Minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 20]
            },
            'min_samples_split': {
                'description': 'Minimum number of samples required to split an internal node.',
                'default': 2,
                'range': [2, 50]
            },
            'max_features': {
                'description': textwrap.dedent('''\
                    The number of features to consider when looking for the best split.
                    Use a float to specify a fraction of features.'''),
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'criterion': {
                'description': 'Function to measure the quality of a split.',
                'default': 'gini',
                'categorical': ['gini', 'entropy', 'log_loss']
            },
            'splitter': {
                'description': 'Strategy used to choose the split at each node.',
                'default': 'best',
                'categorical': ['best', 'random']
            },
            'class_weight': {
                'description': textwrap.dedent('''\
                    The "balanced" mode uses values of y to adjust weights inversely
                    proportional to class frequencies.'''),
                'default': None,
                'categorical': [None, 'balanced']
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: DecisionTreeClassifier = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = DecisionTreeClassifier(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def predict_proba(self, X):
        return super().predict_proba(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
