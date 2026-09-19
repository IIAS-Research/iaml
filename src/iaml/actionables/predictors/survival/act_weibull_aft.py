"""[STEP] Weibull AFT"""
import inspect
import textwrap
from typing import Any

import numpy as np

try:
    from sksurv.linear_model import WeibullAFT
except Exception:  # pragma: no cover - optional dependency
    try:
        from sksurv.parametric import WeibullAFT
    except Exception:  # pragma: no cover - optional dependency
        WeibullAFT = None

try:
    from lifelines import WeibullAFTFitter
except Exception:  # pragma: no cover - optional dependency
    WeibullAFTFitter = None

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActWeibullAFT(Predictor):
    """[STEP] Weibull AFT"""

    name: str = "WeibullAFT"
    _usage: str = "Use when you want a parametric Weibull AFT model with time ratios, rather than ActCox. Applicable to tabular right-censored survival data with numeric features. Avoid when Weibull fit is implausible or you need flexible nonparametric models like ActExtraSurvivalTrees."
    _description: str = textwrap.dedent('''\
        WeibullAFT is a parametric accelerated failure time model
        that assumes survival times follow a Weibull distribution and
        models how covariates speed up or slow down the event time.
        Uses scikit-survival when available and falls back to lifelines.''')
    _description_long: str = textwrap.dedent('''\
        WeibullAFT fits an accelerated failure time model where the log of
        survival time is a linear function of the input features and the
        baseline survival follows a Weibull distribution. The model provides
        parametric survival and hazard estimates, supports right-censored data,
        and yields interpretable covariate effects on time-to-event outcomes.
        This step uses scikit-survival when available and falls back to
        lifelines when needed.''')

    def __init__(self):
        self.configuration: dict = {
            'alpha': {
                'description': textwrap.dedent('''\
                    Regularization strength for scikit-survival. Higher values
                    enforce stronger regularization.'''),
                'default': 0.05,
                'range': [1e-04, 10.0]
            },
            'penalizer': {
                'description': textwrap.dedent('''\
                    L2 penalizer strength for lifelines models.'''),
                'default': 0.0,
                'range': [0.0, 10.0]
            },
            'l1_ratio': {
                'description': textwrap.dedent('''\
                    Mixing parameter between L1 and L2 penalties.'''),
                'default': 0.0,
                'range': [0.0, 1.0]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'max_iter': {
                'description': 'Maximum number of iterations for the optimizer.',
                'default': 1000,
                'range': [10, 100000]
            },
            'tol': {
                'description': 'Stopping tolerance.',
                'default': 1e-07,
                'range': [1e-09, 1e-03]
            }
        }
        self.model: Any = None
        self.columns: list[str] = []
        self.backend: str | None = None
        self._lifelines_event_col: str | None = None
        self._lifelines_time_col: str | None = None

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _resolve_backend(self) -> tuple[str | None, Any | None]:
        if WeibullAFT is not None:
            return 'sksurv', WeibullAFT
        if WeibullAFTFitter is not None:
            return 'lifelines', WeibullAFTFitter
        return None, None

    def _model_parameters(self, model_cls: Any, backend: str) -> dict[str, Any]:
        params = self.passthrough_parameters()
        if backend == 'lifelines':
            params.pop('alpha', None)

        if model_cls is None:
            return params

        try:
            sig_params = inspect.signature(model_cls).parameters
        except (TypeError, ValueError):
            return params

        return {key: value for key, value in params.items() if key in sig_params}

    @staticmethod
    def _split_survival_target(y) -> tuple[np.ndarray, np.ndarray]:
        samples = Dataset.normalize_survival_target(y)
        if not samples:
            return np.array([], dtype=bool), np.array([], dtype=float)
        events, times = zip(*samples)
        return np.asarray(events, dtype=bool), np.asarray(times, dtype=float)

    @staticmethod
    def _unique_column_name(base: str, columns) -> str:
        name = base
        while name in columns:
            name = f"_{name}"
        return name

    def _prepare_lifelines_frame(self, X, y) -> tuple[Any, str, str]:
        X_selected = self._select_features(X).copy()
        event_col = self._unique_column_name("_event", X_selected.columns)
        time_col = self._unique_column_name("_time", X_selected.columns)
        events, times = self._split_survival_target(y)
        X_selected[event_col] = events
        X_selected[time_col] = times
        return X_selected, time_col, event_col

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        backend, model_cls = self._resolve_backend()
        if model_cls is None:
            raise ImportError(
                "WeibullAFT requires scikit-survival or lifelines to be installed."
            )

        self.backend = backend
        self.model = model_cls(**self._model_parameters(model_cls, backend))

        if backend == 'sksurv':
            X, y = dataset.to_survival()
            self.model.fit(self._select_features(X), y)
        else:
            X, time_col, event_col = self._prepare_lifelines_frame(dataset.X, dataset.y)
            self.model.fit(X, duration_col=time_col, event_col=event_col)
            self._lifelines_time_col = time_col
            self._lifelines_event_col = event_col

        return self

    def predict(self, X):
        X_selected = self._select_features(X)
        if self.model and hasattr(self.model, 'predict'):
            return super().predict(X_selected)
        if self.model and hasattr(self.model, 'predict_median'):
            return self.model.predict_median(X_selected)
        if self.model and hasattr(self.model, 'predict_expectation'):
            return self.model.predict_expectation(X_selected)
        return None

    def predict_survival_function(self, X):
        X_selected = self._select_features(X)
        if self.model and hasattr(self.model, 'predict_survival_function'):
            return self.model.predict_survival_function(X_selected)
        raise AttributeError("Unable to predict survival function with this model")

    def predict_cumulative_hazard_function(self, X):
        X_selected = self._select_features(X)
        if self.model and hasattr(self.model, 'predict_cumulative_hazard_function'):
            return self.model.predict_cumulative_hazard_function(X_selected)
        if self.model and hasattr(self.model, 'predict_cumulative_hazard'):
            return self.model.predict_cumulative_hazard(X_selected)
        raise AttributeError("Unable to predict cumulative hazard function with this model")

    def score(self, X, y=None, *args, **kwargs):
        if self.backend == 'lifelines':
            if y is None:
                if (
                    self._lifelines_event_col is None
                    or self._lifelines_time_col is None
                    or not hasattr(X, 'columns')
                    or self._lifelines_event_col not in X.columns
                    or self._lifelines_time_col not in X.columns
                ):
                    raise ValueError(
                        "lifelines score requires y or a DataFrame containing the "
                        "duration/event columns from fit."
                    )
                X_frame = self._select_features(X).copy()
                X_frame[self._lifelines_event_col] = X[self._lifelines_event_col]
                X_frame[self._lifelines_time_col] = X[self._lifelines_time_col]
                return self.model.score(X_frame, *args, **kwargs)
            X_frame = self._select_features(X).copy()
            events, times = self._split_survival_target(y)
            if self._lifelines_event_col is None or self._lifelines_time_col is None:
                self._lifelines_event_col = self._unique_column_name(
                    "_event", X_frame.columns
                )
                self._lifelines_time_col = self._unique_column_name(
                    "_time", X_frame.columns
                )
            X_frame[self._lifelines_event_col] = events
            X_frame[self._lifelines_time_col] = times
            return self.model.score(X_frame, *args, **kwargs)
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        backend_available = WeibullAFT is not None or WeibullAFTFitter is not None
        return dataset.type_of_target == 'survival' \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC)) \
            and backend_available

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
