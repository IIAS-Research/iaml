"""[PLOT] Histogram plot for descriptive statistics."""
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


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


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
    return column_str


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


def _histogram_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size == 0:
        return None
    try:
        numeric = values.astype(float)
    except (ValueError, TypeError):
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


class HistogramPlot(StatisticPlot):
    """[PLOT] Histogram Plot."""

    name: str = "Histogram Plot"
    _description: str = textwrap.dedent("""\
        Histograms show distributions for numeric columns.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot renders a histogram per numeric column to visualize the distribution
        of values. It relies on precomputed histogram statistics when available.
        """)
    refs: list[dict] = []

    title: str = "Histogram plot"
    description: str = textwrap.dedent("""\
        The histogram plot displays distributions for numeric columns.
        """)
    group_by_feature: bool = True

    def __str__(self) -> str:
        return 'histogram'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        dataset: Dataset | None = None,
        **kwargs) -> 'HistogramPlot':
        """Compute histogram plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        hist_key = str(self)
        if hist_key not in dataframe.index:
            _plot_placeholder("Histogram statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        if dataset is not None:
            numeric_columns = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
            columns_to_show = [col for col in dataframe.columns if col in numeric_columns]
        else:
            columns_to_show = list(dataframe.columns)

        histogram_row = dataframe.loc[hist_key]
        entries: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        for col in columns_to_show:
            hist_data = _extract_hist_data(histogram_row.get(col))
            if hist_data is not None:
                entries.append((col, hist_data))

        if not entries:
            _plot_placeholder("No numeric histogram statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        n_plots = len(entries)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes_list = np.atleast_1d(axes).ravel()

        for ax, (col, (counts, bins)) in zip(axes_list, entries):
            if bins.size != counts.size + 1:
                ax.text(0.5, 0.5, "Invalid histogram data", ha='center', va='center')
                ax.axis('off')
                continue
            widths = np.diff(bins)
            ax.bar(bins[:-1], counts, width=widths, align='edge', edgecolor='black')
            ax.set_title(_column_label(col, base_name))
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')

        for ax in axes_list[len(entries):]:
            ax.axis('off')

        if base_name:
            fig.suptitle(f"Histogram: {base_name}")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()

        plt.savefig(self._binary_image, format='png')
        return self
