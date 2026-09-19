"""[STATISTIC] Time Summary."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

from ..dataset import Dataset
from ..statistic import Statistic


class TimeSummaryStatistic(Statistic):
    """[STATISTIC] Time Summary."""

    name: str = "Time Summary"
    _description: str = textwrap.dedent("""\
        Time summary reports key statistics of survival times.
        """)
    _description_long: str = textwrap.dedent("""\
        Time summary reports min/median/max and selected quantiles of survival times.
        """)
    refs: list[dict] = []

    quantiles = [0.1, 0.25, 0.75, 0.9]

    def __str__(self) -> str:
        return 'time_summary'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute survival time summary statistics."""
        if dataset.type_of_target != 'survival':
            return pd.DataFrame()

        samples = Dataset.normalize_survival_target(dataset.y)
        times = np.asarray([time for _, time in samples], dtype=float)

        columns = ['time_min', 'time_median', 'time_max']
        columns += [f"time_quantile_{quantile}" for quantile in self.quantiles]

        if times.size:
            time_min = float(np.min(times))
            time_median = float(np.median(times))
            time_max = float(np.max(times))
            quantile_values = np.quantile(times, self.quantiles).astype(float).tolist()
        else:
            time_min = None
            time_median = None
            time_max = None
            quantile_values = [None] * len(self.quantiles)

        data = [[time_min, time_median, time_max] + quantile_values]
        return pd.DataFrame(data, index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target == 'survival'
