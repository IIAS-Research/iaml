"""Breslow survival curves for Cox log-risk predictions."""
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_consistent_length
from sksurv.functions import StepFunction


class CoxBreslowBaseline:
    """Estimate baseline cumulative hazard from training times and log risks.

    Risk sets include all observations still followed at an event time,
    including those censored at that same time. Computations stay in log space
    to avoid overflowing risks or subtracting nearly equal risk-set sums.
    """

    def __init__(self):
        self.unique_times_ = None
        self._curve_times = None
        self._log_cumulative_hazard = None

    @staticmethod
    def _check_margins(margins):
        margins = np.asarray(margins, dtype=float)
        if margins.ndim != 1 or not np.isfinite(margins).all():
            raise ValueError("Cox survival curves require finite, one-dimensional log risks.")
        return margins

    def fit(self, margins, events, times):
        """Fit Breslow on original, unsigned training durations only."""
        margins = self._check_margins(margins)
        events = np.asarray(events, dtype=bool)
        times = np.asarray(times, dtype=float)
        if events.ndim != 1 or times.ndim != 1:
            raise ValueError("Cox baseline requires one-dimensional events and durations.")
        check_consistent_length(margins, events, times)
        if not events.any():
            raise ValueError("Cox baseline requires at least one observed event.")
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("Cox baseline requires finite, non-negative durations.")

        order = np.argsort(times, kind='mergesort')
        unique_times, first = np.unique(times[order], return_index=True)
        event_counts = np.add.reduceat(events[order].astype(int), first)
        # A reverse accumulation sums each complete risk set directly. Every
        # member of a tie shares its denominator, regardless of censoring.
        log_risk_sets = np.logaddexp.accumulate(margins[order][::-1])[::-1][first]
        log_jumps = np.full(unique_times.shape, -np.inf)
        observed = event_counts > 0
        log_jumps[observed] = np.log(event_counts[observed]) - log_risk_sets[observed]
        log_cumulative_hazard = np.logaddexp.accumulate(log_jumps)

        self.unique_times_ = unique_times
        self._curve_times = unique_times
        # StepFunction clips to its first node. An explicit origin prevents a
        # positive first event from decreasing survival prematurely at t=0.
        if unique_times[0] > 0:
            self._curve_times = np.r_[0.0, unique_times]
            log_cumulative_hazard = np.r_[-np.inf, log_cumulative_hazard]
        self._log_cumulative_hazard = log_cumulative_hazard
        return self

    def _functions(self, margins, survival):
        if self._log_cumulative_hazard is None:
            raise NotFittedError("Fit the Cox baseline before requesting survival curves.")
        margins = self._check_margins(margins)
        # Overflow represents an infinite cumulative hazard, whose survival
        # probability is correctly zero. Adding log risks before exponentiating
        # also keeps curves invariant to an arbitrary Cox intercept shift.
        with np.errstate(over='ignore', under='ignore'):
            values = np.exp(margins[:, None] + self._log_cumulative_hazard[None, :])
            if survival:
                values = np.exp(-values)
        return np.asarray([StepFunction(self._curve_times, row) for row in values], dtype=object)

    def get_survival_function(self, margins):
        """Return survival StepFunctions over [0, max(training duration)]."""
        return self._functions(margins, survival=True)

    def get_cumulative_hazard_function(self, margins):
        """Return cumulative hazard StepFunctions on the training horizon."""
        return self._functions(margins, survival=False)
