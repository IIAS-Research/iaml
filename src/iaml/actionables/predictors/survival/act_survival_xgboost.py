"""XGBoost Cox predictor and the historical scikit-survival import.

``ActGradientBoostingSurvivalAnalysis`` remains the exact same class as in its
canonical module, so existing imports and pickles keep resolving.
"""
import textwrap

import numpy as np
from sklearn.exceptions import NotFittedError
from sksurv.metrics import concordance_index_censored
from xgboost import XGBRegressor

from .._xgboost import xgboost_features
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step
from ....predictor import Predictor
from ._cox_breslow import CoxBreslowBaseline
from .act_gradient_boosting_survival_analysis import ActGradientBoostingSurvivalAnalysis

__all__ = ['ActSurvivalXGBoost', 'ActGradientBoostingSurvivalAnalysis']


def _survival_arrays(y):
    """Normalize IAML targets and reject invalid Cox training/scoring data."""
    samples = Dataset.normalize_survival_target(y)
    if not samples:
        raise ValueError("XGBoost Cox requires non-empty survival targets.")
    events = np.asarray([event for event, _ in samples], dtype=bool)
    times = np.asarray([time for _, time in samples], dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("XGBoost Cox requires finite, non-negative survival durations.")
    if not events.any():
        raise ValueError("XGBoost Cox requires at least one observed event.")
    return events, times


def _cox_labels(events, times):
    """Encode events as positive and right-censored observations as negative.

    Cox uses only the ordering and ties of absolute labels. Replace zero times
    with a common positive value below all positive times, since XGBoost tests
    ``label > 0`` for events. If float32 would lose ordering, ties or signs,
    use positive time ranks instead. Public targets retain their original times.
    """
    unique_times, ranks = np.unique(times, return_inverse=True)
    durations = times.copy()
    with np.errstate(over='ignore', under='ignore'):
        positive = times[times > 0]
        zero_time = min(1.0, positive.min() / 2) if positive.size else 1.0
        durations[times == 0] = zero_time
        magnitudes = durations.astype(np.float32)

    if (not np.isfinite(magnitudes).all() or np.any(magnitudes <= 0)
            or np.unique(magnitudes).size != unique_times.size):
        # Consecutive integers through 2**24 are exactly representable in float32.
        if unique_times.size > 2**24:
            raise ValueError("Too many distinct survival durations for XGBoost Cox labels.")
        magnitudes = (ranks + 1).astype(np.float32)

    return np.where(events, magnitudes, -magnitudes)


@is_step('predictor', 'tabular', 'survival', 'minimal_predictor')
class ActSurvivalXGBoost(Predictor):
    """[STEP] XGBoost Cox proportional hazards with right-censored targets."""

    name: str = 'XGBoost Cox'
    _description: str = "Regularized XGBoost trees trained with the Cox survival objective."
    _description_long: str = textwrap.dedent('''\
        Uses XGBRegressor(objective="survival:cox") to model non-linear covariate
        effects under proportional hazards. Event times are positive labels and
        right-censored times are negative labels, with zero times safely adapted.
        Predictions are hazard ratios: larger scores imply higher event risk.
        Breslow's estimator uses the training log risks and original durations
        to estimate baseline cumulative hazard and individual survival curves.
        Curves are available over the training follow-up horizon, including
        time zero. score() returns Harrell's C-index.''')
    _usage: str = (
        "Use for non-linear Cox survival modeling with numeric or encoded tabular features "
        "and right-censoring. Evaluate risk ranking with a C-index and survival "
        "probabilities with Brier scores; compare predicted and observed survival curves."
    )
    refs: list[dict] = [{
        'name': 'XGBoost: A Scalable Tree Boosting System',
        'year': 2016,
        'authors': ['Tianqi Chen', 'Carlos Guestrin'],
        'doi': 'https://doi.org/10.1145/2939672.2939785',
        'publisher': 'ACM SIGKDD 2016, pages 785--794',
    }]

    def __init__(self):
        self.configuration = {
            'n_estimators': {
                'description': 'Number of boosting trees.',
                'default': 100,
                'range': [1, 500],
            },
            'max_depth': {
                'description': 'Maximum depth of each tree.',
                'default': 3,
                'range': [1, 16],
            },
            'learning_rate': {
                'description': 'Shrinkage applied to each boosting tree.',
                'default': 0.1,
                'range': [0.001, 1.0],
            },
            'subsample': {
                'description': 'Fraction of training rows sampled for each tree.',
                'default': 1.0,
                'range': [0.1, 1.0],
            },
            'colsample_bytree': {
                'description': 'Fraction of feature columns sampled for each tree.',
                'default': 1.0,
                'range': [0.1, 1.0],
            },
            'min_child_weight': {
                'description': 'Minimum sum of instance Hessians required in a child.',
                'default': 1.0,
                'range': [0.0, 20.0],
            },
            'reg_alpha': {
                'description': 'L1 regularization strength on leaf weights.',
                'default': 0.0,
                'range': [0.0, 100.0],
            },
            'reg_lambda': {
                'description': 'L2 regularization strength on leaf weights.',
                'default': 1.0,
                'range': [0.0, 100.0],
            },
            'gamma': {
                'description': 'Minimum loss reduction required to split a leaf.',
                'default': 0.0,
                'range': [0.0, 20.0],
            },
            'random_state': {
                'description': 'Random seed for row and feature sampling.',
                'default': 42,
            },
        }
        self.model: XGBRegressor = None
        self._baseline_model: CoxBreslowBaseline = None

    def fit(self, dataset: Dataset):
        events, times = _survival_arrays(dataset.y)
        if len(times) != len(dataset.X):
            raise ValueError("XGBoost Cox requires one survival target per feature row.")
        self._baseline_model = None
        self.model = XGBRegressor(
            **self.passthrough_parameters(),
            objective='survival:cox',
            tree_method='hist',
            n_jobs=1,
        )
        features = xgboost_features(dataset.X)
        self.model.fit(features, _cox_labels(events, times))
        margins = self.model.predict(features, output_margin=True)
        self._baseline_model = CoxBreslowBaseline().fit(margins, events, times)
        return self

    def predict(self, X):
        """Return hazard ratios; higher values mean higher event risk."""
        return super().predict(xgboost_features(X))

    @property
    def unique_times_(self):
        """Original training follow-up times, sorted and deduplicated."""
        self._require_baseline()
        return self._baseline_model.unique_times_

    def _require_baseline(self):
        if getattr(self, '_baseline_model', None) is None:
            raise NotFittedError("Fit XGBoost Cox before requesting survival curves.")

    def predict_survival_function(self, X):
        """Return Breslow survival curves over [0, max(training duration)]."""
        self._require_baseline()
        margins = self.model.predict(xgboost_features(X), output_margin=True)
        return self._baseline_model.get_survival_function(margins)

    def predict_cumulative_hazard_function(self, X):
        """Return cumulative hazard curves on the original training time scale."""
        self._require_baseline()
        margins = self.model.predict(xgboost_features(X), output_margin=True)
        return self._baseline_model.get_cumulative_hazard_function(margins)

    def score(self, X, y):  # pylint: disable=arguments-differ
        """Return Harrell's C-index on original, unsigned survival targets."""
        events, times = _survival_arrays(y)
        return concordance_index_censored(events, times, self.predict(X))[0]

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
