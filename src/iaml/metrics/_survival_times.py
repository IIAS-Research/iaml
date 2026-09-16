"""Evaluation times shared by the survival Brier metrics."""
import numpy as np


def brier_evaluation_times(durations: np.ndarray) -> np.ndarray:
    """Return up to 100 evenly spaced times in [min(durations), max(durations)).

    A relative grid is independent of the time unit. The endpoint is excluded
    to keep evaluations inside the follow-up interval, including when the last
    observation is censored.
    """
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0 or not np.isfinite(durations).all():
        raise ValueError("Brier scores require non-empty, finite follow-up times.")

    start, stop = durations.min(), durations.max()
    if start >= stop:
        raise ValueError("Brier scores require at least two distinct follow-up times.")

    times = np.linspace(start, stop, num=100, endpoint=False)
    # Very narrow intervals can produce duplicates or round up to the endpoint.
    return np.unique(times[times < stop])
