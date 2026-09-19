"""[STEP] Poisson Regressor"""
import textwrap
from typing import Any

import numpy as np
from sklearn.linear_model import PoissonRegressor

from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActPoissonRegressor(Predictor):
    """[STEP] Poisson Regressor"""

    name: str = "Poisson Regressor"
    _description: str = textwrap.dedent('''\
        PoissonRegressor models count targets using a log link and a Poisson
        likelihood to produce positive predictions.''')
    _description_long: str = textwrap.dedent('''\
        Poisson regression is a generalized linear model designed for
        non-negative count data. It connects predictors to the expected
        count through a log link, which keeps predictions positive and
        is appropriate when variance grows with the mean.''')
    _usage: str = "Use when modeling non-negative count targets with variance rising with mean, as a simpler option than ActElasticNetRegressor or ActDecisionTreeRegressor. Applicable to tabular numeric features with count outcomes. Avoid when targets are continuous, negative, or highly zero-inflated."
    refs: list[dict[str, Any]] = [
        {
            'year': 1972,
            'name': 'Generalized Linear Models',
            'authors': [
                'John A. Nelder',
                'Robert W. M. Wedderburn'
            ],
            'doi': 'https://doi.org/10.2307/2344614',
            'publisher': 'Journal of the Royal Statistical Society, Series A'
        }
    ]

    def __init__(self):
        self.configuration = {
            'alpha': {
                'description': 'L2 regularization strength.',
                'default': 1.0,
                'range': [1e-06, 10.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'max_iter': {
                'description': 'Maximum number of iterations.',
                'default': 100,
                'range': [50, 2000]
            },
            'tol': {
                'description': 'Stopping criterion.',
                'default': 0.0001,
                'range': [1e-06, 0.1]
            },
            'warm_start': {
                'description': 'Reuse solution from the previous fit.',
                'default': False,
                'categorical': [True, False]
            }
        }
        self.model: PoissonRegressor = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _is_count_target(self, y) -> bool:
        y_array = np.asarray(y)
        if y_array.size == 0:
            return False
        if not np.issubdtype(y_array.dtype, np.number):
            return False
        if not np.isfinite(y_array).all():
            return False
        if (y_array < 0).any():
            return False
        return np.allclose(y_array, np.round(y_array), rtol=0, atol=1e-06)

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = PoissonRegressor(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous' \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC)) \
            and self._is_count_target(dataset.y)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
