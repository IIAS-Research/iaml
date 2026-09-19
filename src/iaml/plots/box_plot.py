"""[PLOT] Box plot for descriptive statistics."""
from __future__ import annotations

import io
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..data_type import DataType
from ..dataset import Dataset
from ..plot import StatisticPlot, capture


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _infer_base_name(columns: list[str]) -> str:
    for col in columns:
        if isinstance(col, str) and col.endswith('_all'):
            return col[:-4]
    first = columns[0] if columns else ''
    first_str = str(first)
    return first_str.rsplit('_', 1)[0] if '_' in first_str else first_str


def _column_label(column: str, base_name: str | None) -> str:
    column_str = str(column)
    base_name_str = str(base_name) if base_name is not None else None
    if base_name_str:
        if column_str == base_name_str:
            return 'all'
        prefix = f"{base_name_str}_"
        if column_str.startswith(prefix):
            return column_str[len(prefix):]
    if column_str.endswith('_all'):
        return 'all'
    return column_str.rsplit('_', 1)[-1] if '_' in column_str else column_str


def _base_from_column(column: str) -> str:
    column_str = str(column)
    if column_str.endswith('_all'):
        return column_str[:-4]
    return column_str.rsplit('_', 1)[0] if '_' in column_str else column_str


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


def _extract_box_stats(dataframe: pd.DataFrame, column: str) -> dict[str, float] | None:
    required_rows = {
        'min': 'whislo',
        'max': 'whishi',
        'quantile_0.25': 'q1',
        'quantile_0.5': 'med',
        'quantile_0.75': 'q3',
    }
    stats: dict[str, float] = {}
    for row, key in required_rows.items():
        if row not in dataframe.index:
            return None
        value = dataframe.at[row, column]
        if _is_missing(value):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        stats[key] = value
    stats['fliers'] = []
    return stats


class BoxPlot(StatisticPlot):
    """[PLOT] Box Plot."""

    name: str = "Box Plot"
    _description: str = textwrap.dedent("""\
        Box plots summarize numeric distributions per class.
        """)
    _description_long: str = textwrap.dedent("""\
        Box plots visualize min, quartiles, and max values for numeric columns
        per class, using precomputed descriptive statistics.
        """)
    refs: list[dict] = []

    title: str = "Box plot"
    description: str = textwrap.dedent("""\
        The box plot shows numerical distributions per class.
        """)
    group_by_feature: bool = True

    def __str__(self) -> str:
        return 'boxplot'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        dataset: Dataset | None = None,
        **kwargs) -> 'BoxPlot':
        """Compute box plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        base_name = base_name or _infer_base_name(list(dataframe.columns))

        if dataset is not None:
            numeric_columns = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
            if base_name in numeric_columns:
                columns_to_show = [
                    col for col in dataframe.columns
                    if _base_from_column(col) == base_name
                ]
            else:
                columns_to_show = [
                    col for col in dataframe.columns
                    if _base_from_column(col) in numeric_columns
                ]
        else:
            columns_to_show = list(dataframe.columns)

        if not columns_to_show:
            _plot_placeholder("No numeric statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        entries: list[dict[str, Any]] = []
        for col in columns_to_show:
            stats = _extract_box_stats(dataframe, col)
            if stats is None:
                continue
            stats['label'] = _column_label(col, base_name)
            entries.append(stats)

        if not entries:
            _plot_placeholder("No box plot statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        plt.figure(figsize=(max(4.0, 0.9 * len(entries)), 4.0))
        plt.bxp(entries, showfliers=False)
        plt.ylabel('Value')
        if base_name:
            plt.title(f"Box plot: {base_name}")
        plt.tight_layout()
        plt.savefig(self._binary_image, format='png')
        return self
