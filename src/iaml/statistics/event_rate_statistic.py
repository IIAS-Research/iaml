"""[STATISTIC] Event Rate."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..statistic import Statistic


class EventRateStatistic(Statistic):
    """[STATISTIC] Event Rate."""

    name: str = "Event Rate"
    _description: str = textwrap.dedent("""\
        Event rate measures the proportion of events versus censoring in survival data.
        """)
    _description_long: str = textwrap.dedent("""\
        Event rate measures the proportion of events versus censoring in survival targets.
        Rates are reported as fractions in [0, 1].
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'event_rate'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute event and censoring rates for survival targets."""
        if dataset.type_of_target != 'survival':
            return pd.DataFrame()

        samples = Dataset.normalize_survival_target(dataset.y)
        n_samples = len(samples)

        if n_samples:
            event_count = int(sum(event for event, _ in samples))
            censor_count = n_samples - event_count
            event_rate = event_count / n_samples
            censor_rate = censor_count / n_samples
        else:
            event_count = 0
            censor_count = 0
            event_rate = 0.0
            censor_rate = 0.0

        data = [[event_count, censor_count, event_rate, censor_rate]]
        columns = ['event_count', 'censor_count', 'event_rate', 'censor_rate']
        return pd.DataFrame(data, index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target == 'survival'
