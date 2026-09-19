"""[STEP] Huber Regressor"""
import textwrap
from typing import Any
from sklearn.linear_model import HuberRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActHuberRegressor(Predictor):
    """[STEP] Huber Regressor"""

    name: str = "Huber Regressor"
    _description: str = textwrap.dedent('''\
        HuberRegressor fits a linear model that is less sensitive to outliers
        by combining squared and absolute error losses.''')
    _description_long: str = textwrap.dedent('''\
        HuberRegressor optimizes a loss that behaves like squared error for
        small residuals and like absolute error for large residuals. This
        makes it a robust choice for tabular regression when data can contain
        outliers while retaining efficiency for clean data.''')
    _usage: str = "Use when you need robust linear regression with outliers; prefer over ActElasticNetRegressor when outliers skew fits. Applicable to numeric tabular regression with mostly linear relationships. Avoid when nonlinear effects or categorical-heavy data suit ActCatBoostRegressor."
    refs: list[dict[str, Any]] = [
        {
            'year': 1964,
            'name': 'Robust Estimation of a Location Parameter',
            'authors': [
                'Peter J. Huber'
            ],
            'doi': 'https://doi.org/10.1214/aoms/1177703732',
            'publisher': 'Annals of Mathematical Statistics'
        }
    ]

    def __init__(self):
        self.configuration = {
            'epsilon': {
                'description': textwrap.dedent('''\
                    Threshold that controls the point where the loss switches
                    from quadratic to linear.'''),
                'default': 1.35,
                'range': [1.01, 3.0]
            },
            'alpha': {
                'description': 'L2 regularization strength.',
                'default': 0.0001,
                'range': [1e-07, 1.0]
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
                'default': 1e-05,
                'range': [1e-06, 0.1]
            },
            'warm_start': {
                'description': 'Reuse solution from previous fit as initialization.',
                'default': False,
                'categorical': [True, False]
            }
        }
        self.model: HuberRegressor = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = HuberRegressor(**self.passthrough_parameters())
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
