"""[STEP] Fast Survival SVM"""
import inspect
import textwrap
from typing import Any
from sksurv.svm import FastSurvivalSVM

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActFastSurvivalSVM(Predictor):
    """[STEP] Fast Survival SVM"""

    name: str = "FastSurvivalSVM"
    _usage: str = "Use when you want a fast linear ranking model for survival risk, as a simpler alternative to ActCox. Applicable to right-censored tabular survival data with mostly numeric features. Avoid when nonlinear effects or interactions dominate; prefer ActRandomSurvivalForest."
    _description: str = textwrap.dedent('''\
        FastSurvivalSVM is a linear support vector machine for survival analysis
        that learns a risk score using hinge-style ranking losses adapted to
        censored data.''')
    _description_long: str = textwrap.dedent('''\
        FastSurvivalSVM optimizes a pairwise ranking objective so that samples
        with earlier events receive higher risk scores. The loss is based on
        hinge-style constraints adapted to right-censored observations, and a
        rank_ratio parameter can blend ranking and regression terms. The model
        is linear and efficient, making it suitable for larger tabular datasets
        where fast, deterministic training is desired.''')

    def __init__(self):
        self.configuration: dict = {
            'alpha': {
                'description': textwrap.dedent('''\
                    Regularization strength. Higher values enforce stronger
                    regularization.'''),
                'default': 1.0,
                'range': [1e-04, 100.0]
            },
            'rank_ratio': {
                'description': textwrap.dedent('''\
                    Weighting between ranking and regression losses. 1.0 uses
                    pure ranking; 0.0 uses pure regression.'''),
                'default': 1.0,
                'range': [0.0, 1.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'max_iter': {
                'description': 'Maximum number of iterations for the optimizer.',
                'default': 200,
                'range': [10, 5000]
            },
            'tol': {
                'description': 'Stopping tolerance.',
                'default': 1e-05,
                'range': [1e-08, 1e-02]
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: FastSurvivalSVM = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _model_parameters(self) -> dict[str, Any]:
        params = self.passthrough_parameters()
        sig_params = inspect.signature(FastSurvivalSVM).parameters
        return {key: value for key, value in params.items() if key in sig_params}

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = FastSurvivalSVM(**self._model_parameters())
        X, y = dataset.to_survival()
        self.model.fit(self._select_features(X), y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival' \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
