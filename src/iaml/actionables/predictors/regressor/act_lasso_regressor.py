"""[STEP] Lasso Regressor"""
import textwrap
from typing import Any
from sklearn.linear_model import Lasso
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActLassoRegressor(Predictor):
    """[STEP] Lasso Regressor"""

    name: str = "Lasso Regressor"
    _description: str = textwrap.dedent('''\
        Lasso regression uses L1 regularization to shrink coefficients and
        perform automatic feature selection in linear regression.''')
    _description_long: str = textwrap.dedent('''\
        Lasso (Least Absolute Shrinkage and Selection Operator) fits a linear
        model while adding an L1 penalty to the loss. The penalty drives some
        coefficients to zero, selecting a sparse set of features and improving
        interpretability for tabular regression tasks.''')
    _usage: str = "Use when you want sparse linear coefficients and feature selection; compare ActElasticNetRegressor or ActARDRegression for similar linear shrinkage. Applicable to numeric tabular regression with continuous targets. Avoid when effects are strongly nonlinear or dominated by categorical features."
    refs: list[dict[str, Any]] = [
        {
            'year': 1996,
            'name': 'Regression Shrinkage and Selection via the Lasso',
            'authors': [
                'Robert Tibshirani'
            ],
            'doi': 'https://doi.org/10.1111/j.2517-6161.1996.tb02080.x',
            'publisher': 'Journal of the Royal Statistical Society Series B'
        }
    ]

    def __init__(self):
        self.configuration = {
            'alpha': {
                'description': 'Regularization strength.',
                'default': 1.0,
                'range': [1e-04, 10.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'max_iter': {
                'description': 'Maximum number of iterations.',
                'default': 1000,
                'range': [100, 5000]
            },
            'tol': {
                'description': 'Stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'selection': {
                'description': 'Coordinate descent selection strategy.',
                'default': 'cyclic',
                'categorical': ['cyclic', 'random']
            },
            'positive': {
                'description': 'Force coefficients to be positive.',
                'default': False,
                'categorical': [True, False]
            },
            'random_state': {
                'description': 'Random state used when selection is "random".',
                'default': 42
            }
        }
        self.model: Lasso = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = Lasso(**self.passthrough_parameters())
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
