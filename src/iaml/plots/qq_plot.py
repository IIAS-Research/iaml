"""[PLOT] QQ plot for descriptive statistics."""
from __future__ import annotations

import io
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ..data_type import DataType
from ..dataset import Dataset
from ..plot import StatisticPlot, capture


def _is_missing(value: Any) -> bool:
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


def _sample_values(values: np.ndarray, max_points: int = 2000) -> np.ndarray:
    if values.size <= max_points:
        return values
    indices = np.linspace(0, values.size - 1, max_points, dtype=int)
    return values[indices]


def _sanitize_pair(
    theoretical: np.ndarray,
    ordered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        theoretical_arr = np.asarray(theoretical, dtype=float).ravel()
        ordered_arr = np.asarray(ordered, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if theoretical_arr.size == 0 or ordered_arr.size == 0:
        return None
    if theoretical_arr.size != ordered_arr.size:
        return None
    mask = np.isfinite(theoretical_arr) & np.isfinite(ordered_arr)
    if not np.any(mask):
        return None
    theoretical_arr = theoretical_arr[mask]
    ordered_arr = ordered_arr[mask]
    order = np.argsort(theoretical_arr)
    return theoretical_arr[order], ordered_arr[order]


def _qq_from_sample(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size < 2:
        return None
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return None
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return None
    ordered = np.sort(numeric)
    ordered = _sample_values(ordered)
    n = ordered.size
    if n < 2:
        return None
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs)
    return theoretical, ordered


def _extract_qq_data(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        if 'theoretical' in value and 'ordered' in value:
            return _sanitize_pair(value['theoretical'], value['ordered'])
        if 'theoretical_quantiles' in value and 'sample_quantiles' in value:
            return _sanitize_pair(value['theoretical_quantiles'], value['sample_quantiles'])
        if 'values' in value:
            return _qq_from_sample(np.asarray(value['values']))
        if 'sample' in value:
            return _qq_from_sample(np.asarray(value['sample']))
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            first = np.asarray(value[0])
            second = np.asarray(value[1])
            if first.ndim == 1 and second.ndim == 1 and first.size == second.size:
                return _sanitize_pair(first, second)
        return _qq_from_sample(np.asarray(value))
    return None


class QQPlot(StatisticPlot):
    """[PLOT] QQ Plot."""

    name: str = "QQ Plot"
    _description: str = textwrap.dedent("""\
        QQ plots compare numeric distributions to a normal reference.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot compares ordered sample values against theoretical quantiles
        of the normal distribution, highlighting departures from normality.
        """)
    refs: list[dict] = []

    title: str = "QQ plot"
    description: str = textwrap.dedent("""\
        The QQ plot compares numeric columns to a normal distribution.
        """)
    group_by_feature: bool = True

    def __str__(self) -> str:
        return 'qqplot'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        dataset: Dataset | None = None,
        **kwargs,
    ) -> 'QQPlot':
        """Compute QQ plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        qq_key = str(self)
        if qq_key not in dataframe.index:
            for candidate in ('qq', 'qq_plot'):
                if candidate in dataframe.index:
                    qq_key = candidate
                    break
            else:
                _plot_placeholder("QQ plot statistics not available")
                plt.savefig(self._binary_image, format='png')
                return self

        if dataset is not None:
            numeric_columns = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
            columns_to_show = [col for col in dataframe.columns if col in numeric_columns]
        else:
            columns_to_show = list(dataframe.columns)

        qq_row = dataframe.loc[qq_key]
        entries: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        for col in columns_to_show:
            qq_data = _extract_qq_data(qq_row.get(col))
            if qq_data is not None:
                entries.append((col, qq_data))

        if not entries:
            _plot_placeholder("No numeric QQ statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        n_plots = len(entries)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes_list = np.atleast_1d(axes).ravel()

        for ax, (col, (theoretical, ordered)) in zip(axes_list, entries):
            if theoretical.size == 0 or ordered.size == 0 or theoretical.size != ordered.size:
                ax.text(0.5, 0.5, "Invalid QQ data", ha='center', va='center')
                ax.axis('off')
                continue
            ax.scatter(theoretical, ordered, s=12, alpha=0.7, color='tab:blue')
            mean = float(np.mean(ordered))
            std = float(np.std(ordered, ddof=1)) if ordered.size > 1 else 0.0
            x_min = float(np.min(theoretical))
            x_max = float(np.max(theoretical))
            line_x = np.array([x_min, x_max], dtype=float)
            if np.isfinite(std) and std > 0:
                line_y = mean + std * line_x
            else:
                line_y = np.array([mean, mean], dtype=float)
            ax.plot(line_x, line_y, color='red', linewidth=1)
            ax.set_title(_column_label(col, base_name))
            ax.set_xlabel('Theoretical quantiles')
            ax.set_ylabel('Ordered values')

        for ax in axes_list[len(entries):]:
            ax.axis('off')

        if base_name:
            fig.suptitle(f"QQ plot: {base_name}")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()

        plt.savefig(self._binary_image, format='png')
        return self
