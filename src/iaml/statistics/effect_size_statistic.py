"""[STATISTIC] Effect size (Cohen d)."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class EffectSizeStatistic(Statistic):
    """[STATISTIC] Effect size (Cohen d)."""

    name: str = "Effect Size"
    _description: str = textwrap.dedent("""\
        Effect size measures standardized differences (Cohen d) between classes.
        """)
    _description_long: str = textwrap.dedent("""\
        Effect size computes Cohen's d for each numerical column per class
        compared against the rest of the data. The `_all` suffix reports
        the mean absolute effect size across classes for each feature.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'effect_size'

    def _safe_variance(self, n: float, total: float, total_sq: float) -> float | None:
        if n is None or n <= 1:
            return None
        denom = n - 1.0
        var = (total_sq - (total * total) / n) / denom
        if pd.isna(var) or var < 0:
            return None
        return float(var)

    def _cohen_d(self, mean_a: float, mean_b: float, var_a: float | None,
                 var_b: float | None, n_a: float, n_b: float) -> float | None:
        if var_a is None or var_b is None:
            return None
        if n_a <= 1 or n_b <= 1:
            return None
        pooled_denom = (n_a + n_b - 2.0)
        if pooled_denom <= 0:
            return None
        pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / pooled_denom
        if pooled_var <= 0 or pd.isna(pooled_var):
            return None
        pooled_std = np.sqrt(pooled_var)
        if pooled_std == 0 or pd.isna(pooled_std):
            return None
        return (mean_a - mean_b) / pooled_std

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute Cohen's d for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not numeric_columns:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))

        values = dataset.X[numeric_columns]
        total_n = values.count()
        total_sum = values.sum()
        total_sumsq = (values ** 2).sum()

        grouped = values.groupby(dataset.y)
        class_n = grouped.count()
        class_sum = grouped.sum()
        class_sumsq = (values ** 2).groupby(dataset.y).sum()

        data = []
        columns = []

        for col in numeric_columns:
            col_total_n = float(total_n.get(col, 0))
            col_total_sum = float(total_sum.get(col, 0))
            col_total_sumsq = float(total_sumsq.get(col, 0))
            abs_effects = []

            for label in ['all'] + class_labels:
                columns.append(f"{col}_{label}")
                if label == 'all':
                    data.append(None)
                    continue

                n_a = float(class_n.at[label, col]) if label in class_n.index else 0.0
                if n_a <= 0:
                    data.append(None)
                    continue

                sum_a = float(class_sum.at[label, col]) if label in class_sum.index else 0.0
                sumsq_a = float(class_sumsq.at[label, col]) if label in class_sumsq.index else 0.0

                n_b = col_total_n - n_a
                if n_b <= 0:
                    data.append(None)
                    continue

                sum_b = col_total_sum - sum_a
                sumsq_b = col_total_sumsq - sumsq_a

                mean_a = sum_a / n_a if n_a > 0 else None
                mean_b = sum_b / n_b if n_b > 0 else None
                if mean_a is None or mean_b is None or pd.isna(mean_a) or pd.isna(mean_b):
                    data.append(None)
                    continue

                var_a = self._safe_variance(n_a, sum_a, sumsq_a)
                var_b = self._safe_variance(n_b, sum_b, sumsq_b)
                d_value = self._cohen_d(mean_a, mean_b, var_a, var_b, n_a, n_b)
                data.append(d_value)
                if d_value is not None and not pd.isna(d_value):
                    abs_effects.append(abs(d_value))

            if abs_effects:
                data_index = columns.index(f"{col}_all")
                data[data_index] = float(np.mean(abs_effects))

        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
