"""[PLOT] Target distribution plot for descriptive statistics."""
from __future__ import annotations

import io
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..dataset import Dataset
from ..plot import StatisticPlot, capture


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


def _extract_target_column(dataframe: pd.DataFrame) -> str | None:
    if dataframe.empty:
        return None
    for col in ('target', 'y', 'target_all', 'y_all'):
        if col in dataframe.columns:
            return col
    if len(dataframe.columns) == 1:
        return dataframe.columns[0]
    return None


def _extract_value_counts(value: Any) -> dict[str, float] | None:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        if any(key in value for key in ('counts', 'bins', 'bin_edges', 'hist', 'values')):
            return None
        counts: dict[str, float] = {}
        for key, count in value.items():
            if _is_missing(count):
                continue
            counts[str(key)] = float(count)
        return counts or None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        try:
            items = list(value)
        except TypeError:
            return None
        if not items:
            return None
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in items):
            counts = {}
            for key, count in items:
                if _is_missing(count):
                    continue
                counts[str(key)] = float(count)
            return counts or None
    return None


def _histogram_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size == 0:
        return None
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return None
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return None
    bins = int(np.sqrt(numeric.size))
    bins = max(5, min(20, bins))
    counts, bin_edges = np.histogram(numeric, bins=bins)
    return counts, bin_edges


def _extract_hist_data(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        if 'counts' in value and ('bins' in value or 'bin_edges' in value):
            counts = np.asarray(value['counts'])
            bins = np.asarray(value.get('bins', value.get('bin_edges')))
            return counts, bins
        if 'hist' in value and ('bins' in value or 'bin_edges' in value):
            counts = np.asarray(value['hist'])
            bins = np.asarray(value.get('bins', value.get('bin_edges')))
            return counts, bins
        if 'values' in value:
            return _histogram_from_values(np.asarray(value['values']))
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            first = np.asarray(value[0])
            second = np.asarray(value[1])
            if first.ndim == 1 and second.ndim == 1:
                if first.size == second.size + 1:
                    return second, first
                if second.size == first.size + 1:
                    return first, second
                if first.size == second.size and first.size > 0:
                    bins = np.arange(first.size + 1)
                    return first, bins
        return _histogram_from_values(np.asarray(value))
    return None


def _plot_class_counts(counts: dict[str, float], title: str) -> None:
    plt.figure()
    labels = list(counts.keys())
    values = np.asarray(list(counts.values()), dtype=float)
    x = np.arange(len(labels))
    plt.bar(x, values, color='tab:blue')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()


def _plot_histogram(counts: np.ndarray, bins: np.ndarray, title: str) -> None:
    if bins.size != counts.size + 1:
        _plot_placeholder("Invalid histogram data")
        return
    plt.figure()
    widths = np.diff(bins)
    plt.bar(bins[:-1], counts, width=widths, align='edge', edgecolor='black')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()


class TargetDistributionPlot(StatisticPlot):
    """[PLOT] Target Distribution Plot."""

    name: str = "Target Distribution"
    _description: str = textwrap.dedent("""\
        Target distribution plots summarize the target values.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot shows the distribution of the target variable. For regression tasks,
        it renders a histogram of the target values. For classification tasks, it shows
        counts per class label.
        """)
    refs: list[dict] = []

    title: str = "Target distribution"
    description: str = textwrap.dedent("""\
        The target distribution plot shows the distribution of the target values.
        """)
    group_by_feature: bool = False

    def __str__(self) -> str:
        return 'target_distribution'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        dataset: Dataset | None = None,
        base_name: str | None = None,
        **kwargs,
    ) -> 'TargetDistributionPlot':
        """Compute the target distribution plot."""
        self._binary_image = io.BytesIO()

        title = "Target distribution"
        if base_name:
            title = f"Target distribution: {base_name}"

        if not dataframe.empty:
            target_col = _extract_target_column(dataframe)
            row = None
            row_key = str(self)
            if row_key in dataframe.index:
                row = dataframe.loc[row_key]
            elif 'value_counts' in dataframe.index:
                row = dataframe.loc['value_counts']
            elif 'histogram' in dataframe.index:
                row = dataframe.loc['histogram']

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0] if not row.empty else None

            if row is not None and target_col is not None:
                value = row.get(target_col)
                counts = _extract_value_counts(value)
                if counts:
                    _plot_class_counts(counts, title)
                    plt.savefig(self._binary_image, format='png')
                    return self

                hist = _extract_hist_data(value)
                if hist is not None:
                    counts_arr, bins = hist
                    _plot_histogram(
                        np.asarray(counts_arr, dtype=float),
                        np.asarray(bins, dtype=float),
                        title,
                    )
                    plt.savefig(self._binary_image, format='png')
                    return self

        if dataset is None:
            _plot_placeholder("Target distribution not available")
            plt.savefig(self._binary_image, format='png')
            return self

        if dataset.type_of_target == 'survival':
            _plot_placeholder("Target distribution not available for survival targets")
            plt.savefig(self._binary_image, format='png')
            return self

        y_values = pd.Series(dataset.y)
        if dataset.type_of_target == 'continuous':
            numeric = pd.to_numeric(y_values, errors='coerce').to_numpy()
            numeric = numeric[np.isfinite(numeric)]
            hist = _histogram_from_values(numeric)
            if hist is None:
                _plot_placeholder("No numeric target values available")
            else:
                counts_arr, bins = hist
                _plot_histogram(counts_arr, bins, title)
        else:
            counts_series = y_values.value_counts(dropna=False)
            if counts_series.empty:
                _plot_placeholder("No target labels available")
            else:
                counts = {str(key): float(val) for key, val in counts_series.items()}
                _plot_class_counts(counts, title)

        plt.savefig(self._binary_image, format='png')
        return self
