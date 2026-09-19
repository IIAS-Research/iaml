"""[STEP] Ridge Regressor"""
import textwrap
from typing import Any
from sklearn.linear_model import Ridge
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActRidgeRegressor(Predictor):
    """[STEP] Ridge Regressor"""

    name: str = "Ridge Regressor"
    _description: str = textwrap.dedent('''\
        Ridge regression applies L2 regularization to stabilize coefficients
        when predictors are correlated.''')
    _description_long: str = textwrap.dedent('''\
        Ridge regression fits a linear model while penalizing large coefficients
        with an L2 term. This reduces variance, improves numerical stability,
        and provides robust predictions for tabular regression problems.''')
    _usage: str = "Use when you need a stable linear regressor for correlated numeric features; compare ActElasticNetRegressor or ActARDRegression. Applicable to tabular regression with mostly numeric inputs. Avoid when strong nonlinear effects dominate or labels are not continuous."
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
            'random_state': {
                'description': 'Random state for solvers that use randomness.',
                'default': 42
            }
        }
        self.model: Ridge = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = Ridge(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous' \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
