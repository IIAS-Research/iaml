"""[STEP] Elastic Net Regressor"""
import textwrap
from typing import Any
from sklearn.linear_model import ElasticNet
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActElasticNetRegressor(Predictor):
    """[STEP] Elastic Net Regressor"""

    name: str = "Elastic Net Regressor"
    _description: str = textwrap.dedent('''\
        ElasticNet regression combines L1 and L2 regularization to balance
        sparsity and stability in linear models.''')
    _description_long: str = textwrap.dedent('''\
        ElasticNet regression fits a linear model with both L1 and L2 penalties.
        The L1 term encourages sparsity by driving some coefficients to zero,
        while the L2 term stabilizes coefficients when predictors are correlated,
        making it a robust choice for tabular regression.''')
    _usage: str = "Use when you need a sparse linear baseline for tabular regression and want a simpler choice than ActExtraTreesRegressor. Applicable to numeric-feature regression with correlated predictors. Avoid when strong nonlinear patterns dominate; consider ActCatBoostRegressor."
    refs: list[dict[str, Any]] = [
        {
            'year': 2005,
            'name': 'Regularization and Variable Selection via the Elastic Net',
            'authors': [
                'Hui Zou',
                'Trevor Hastie'
            ],
            'doi': 'https://doi.org/10.1111/j.1467-9868.2005.00503.x',
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
            'l1_ratio': {
                'description': 'Mixing parameter between L1 and L2 penalty.',
                'default': 0.5,
                'range': [0.0, 1.0]
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
        self.model: ElasticNet = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = ElasticNet(**self.passthrough_parameters())
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
