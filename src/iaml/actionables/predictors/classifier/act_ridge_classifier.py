"""[STEP] Ridge Classifier"""
import textwrap
from typing import Any
from sklearn.linear_model import RidgeClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActRidgeClassifier(Predictor):
    """[STEP] Ridge Classifier"""

    name: str = "Ridge Classifier"
    _usage: str = "Use when you need a fast linear L2 baseline on numeric tabular data, vs ActDecisionTreeClassifier or ActExtraTreesClassifier. Applicable to binary/multiclass targets with mostly numeric features. Avoid when data is mostly categorical, highly nonlinear, or ActCatBoost fits better."
    _description: str = textwrap.dedent('''\
        RidgeClassifier is a linear classifier that applies L2
        regularization to reduce sensitivity to noisy features.''')
    _description_long: str = textwrap.dedent('''\
        RidgeClassifier fits a linear decision boundary by solving a
        regularized least squares problem. The L2 penalty stabilizes
        coefficients when features are correlated and improves robustness
        to noise in tabular datasets.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1970,
            'name': 'Ridge Regression: Biased Estimation for Nonorthogonal Problems',
            'authors': [
                'Arthur E. Hoerl',
                'Robert W. Kennard'
            ],
            'doi': 'https://doi.org/10.2307/1267351',
            'publisher': 'Technometrics Vol. 12, No. 1, page 55--67'
        }
    ]

    def __init__(self):
        self.configuration = {
            'alpha': {
                'description': 'Regularization strength.',
                'default': 1.0,
                'range': [1e-04, 100.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'solver': {
                'description': 'Solver to use in the ridge optimization.',
                'default': 'auto',
                'categorical': [
                    'auto',
                    'svd',
                    'cholesky',
                    'lsqr',
                    'sparse_cg',
                    'sag',
                    'saga',
                    'lbfgs'
                ]
            },
            'tol': {
                'description': 'Stopping criterion for iterative solvers.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'max_iter': {
                'description': 'Maximum number of iterations for iterative solvers.',
                'default': 1000,
                'range': [50, 5000]
            },
            'class_weight': {
                'description': textwrap.dedent('''\
                    The "balanced" mode uses values of y to adjust weights inversely
                    proportional to class frequencies.'''),
                'default': None,
                'categorical': [None, 'balanced']
            },
            'random_state': {
                'description': 'Random state for solvers that use randomness.',
                'default': 42
            }
        }
        self.model: RidgeClassifier = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = RidgeClassifier(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass', 'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
