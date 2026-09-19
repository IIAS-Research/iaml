"""[STEP] RANSAC Regressor"""
import textwrap
from typing import Any
from sklearn.linear_model import RANSACRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActRANSACRegressor(Predictor):
    """[STEP] RANSAC Regressor"""

    name: str = "RANSAC Regressor"
    _usage: str = "Use when outliers can skew a linear model and you need robustness vs ActElasticNetRegressor. Applicable to numeric tabular regression with moderate sample size. Avoid when data is clean or effects are highly nonlinear; consider ActDecisionTreeRegressor."
    _description: str = textwrap.dedent('''\
        RANSACRegressor fits a robust regression model by iteratively
        sampling subsets of the data and keeping inliers.''')
    _description_long: str = textwrap.dedent('''\
        RANSACRegressor is a robust regression technique that repeatedly fits
        a base estimator on random subsets, identifies inliers using a
        residual threshold, and refits on the consensus set. This approach
        reduces the influence of outliers in tabular regression problems.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1981,
            'name': (
                'Random Sample Consensus: A Paradigm for Model Fitting with '
                'Applications to Image Analysis and Automated Cartography'
            ),
            'authors': [
                'Martin A. Fischler',
                'Robert C. Bolles'
            ],
            'doi': 'https://doi.org/10.1145/358669.358692',
            'publisher': 'Communications of the ACM'
        }
    ]

    def __init__(self):
        self.configuration = {
            'min_samples': {
                'description': textwrap.dedent('''\
                    Minimum number of samples chosen for estimating the model.
                    When None, uses the number of features plus one.'''),
                'default': None
            },
            'residual_threshold': {
                'description': textwrap.dedent('''\
                    Maximum residual for a data sample to be classified as an inlier.
                    When None, uses the median absolute deviation of the target.'''),
                'default': None
            },
            'max_trials': {
                'description': 'Maximum number of iterations for random sampling.',
                'default': 100,
                'range': [10, 500]
            },
            'stop_probability': {
                'description': textwrap.dedent('''\
                    Probability that at least one outlier-free sample has been chosen
                    after max_trials iterations.'''),
                'default': 0.99,
                'range': [0.8, 0.999]
            },
            'loss': {
                'description': 'Loss function used to classify inliers.',
                'default': 'absolute_error',
                'categorical': ['absolute_error', 'squared_error']
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: RANSACRegressor = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = RANSACRegressor(**self.passthrough_parameters())
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
