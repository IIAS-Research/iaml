"""[STEP] Quantile Regressor"""
import textwrap
from typing import Any

from sklearn.linear_model import QuantileRegressor

from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActQuantileRegressor(Predictor):
    """[STEP] Quantile Regressor"""

    name: str = "Quantile Regressor"
    _usage: str = "Use when you need conditional quantiles or asymmetric error control; choose over ActElasticNetRegressor for interval focus. Applicable to tabular regression with continuous targets and numeric inputs. Avoid when mean prediction is enough or nonlinear structure dominates."
    _description: str = textwrap.dedent('''\
        QuantileRegressor estimates a conditional quantile of a continuous target,
        enabling interval-style predictions and asymmetric error handling.''')
    _description_long: str = textwrap.dedent('''\
        Quantile regression models a chosen quantile of the response rather than the
        mean, which makes it useful for prediction intervals and robust modeling.
        By selecting different quantiles, the model can describe the uncertainty
        around the target distribution.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1978,
            'name': 'Regression Quantiles',
            'authors': [
                'Roger Koenker',
                'Gilbert Bassett Jr.'
            ],
            'doi': 'https://doi.org/10.2307/1913643',
            'publisher': 'Econometrica'
        }
    ]

    def __init__(self):
        self.configuration = {
            'quantile': {
                'description': 'Quantile to estimate between 0 and 1.',
                'default': 0.5,
                'range': [0.05, 0.95]
            },
            'alpha': {
                'description': 'L1 regularization strength.',
                'default': 1.0,
                'range': [1e-06, 10.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            }
        }
        self.model: QuantileRegressor = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = QuantileRegressor(**self.passthrough_parameters())
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
