"""[STEP] HistGradient Boosting Classifier"""
import textwrap
from typing import Any
from sklearn.ensemble import HistGradientBoostingClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActHistGradientBoostingClassifier(Predictor):
    """[STEP] HistGradient Boosting Classifier"""

    name: str = "HistGradient Boosting Classifier"
    _description: str = textwrap.dedent('''\
        HistGradientBoostingClassifier is a gradient boosting method that
        uses histogram-based binning to speed up training on large datasets.''')
    _description_long: str = textwrap.dedent('''\
        HistGradientBoostingClassifier builds a sequence of decision trees using
        histograms of feature values to reduce computation. It supports
        regularization and early stopping to control overfitting while keeping
        training fast on large datasets.''')
    _usage: str = "Use when you want strong tabular accuracy on large numeric data; compare ActCatBoost for categorical-heavy data or ActExtraTreesClassifier for simpler ensembles. Applicable to binary, multiclass, or multilabel-indicator targets with numeric features. Avoid when features are mostly categorical or a simple baseline like ActDecisionTreeClassifier is preferred."
    refs: list[dict[str, Any]] = [
        {
            'year': 2017,
            'name': 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
            'authors': [
                'Guolin Ke',
                'Qi Meng',
                'Thomas Finley',
                'Taifeng Wang',
                'Wei Chen',
                'Weidong Ma',
                'Qiwei Ye',
                'Tie-Yan Liu'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1712.01005',
            'publisher': 'Advances in Neural Information Processing Systems 30 (NeurIPS 2017)'
        }
    ]

    def __init__(self):
        self.configuration = {
            'learning_rate': {
                'description': 'The learning rate, also known as shrinkage.',
                'default': 0.1,
                'range': [0.01, 1.0]
            },
            'max_iter': {
                'description': 'The maximum number of boosting iterations.',
                'default': 100,
                'range': [50, 500]
            },
            'max_leaf_nodes': {
                'description': 'The maximum number of leaves for each tree.',
                'default': 31,
                'range': [3, 2048]
            },
            'max_depth': {
                'description': 'The maximum depth of each tree.',
                'default': None,
                'categorical': [None, 3, 5, 10, 15]
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples per leaf.',
                'default': 20,
                'range': [1, 200]
            },
            'l2_regularization': {
                'description': 'The L2 regularization parameter.',
                'default': 1e-10,
                'range': [1e-10, 1.0]
            },
            'max_bins': {
                'description': 'The maximum number of bins to use for continuous features.',
                'default': 255,
                'range': [16, 255]
            },
            'validation_fraction': {
                'description': 'The proportion of data used for early stopping.',
                'default': 0.1,
                'range': [0.05, 0.3]
            },
            'n_iter_no_change': {
                'description': 'Used to determine when to early stop.',
                'default': 10,
                'range': [2, 20]
            },
            'tol': {
                'description': 'The absolute tolerance to use when comparing scores.',
                'default': 1e-4,
                'range': [1e-8, 1e-2]
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
        self.model: HistGradientBoostingClassifier = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = HistGradientBoostingClassifier(
            early_stopping=True,
            **self.passthrough_parameters()
        )
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
            ['binary', 'multiclass', 'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
