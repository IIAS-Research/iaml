"""[PLOT] Density plot for descriptive statistics."""
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


def _kde_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size < 2:
        return None
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return None
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return None

    sample = _sample_values(numeric)
    vmin = float(np.min(sample))
    vmax = float(np.max(sample))
    if vmin == vmax:
        eps = 1e-3 if abs(vmin) < 1 else abs(vmin) * 1e-3
        support = np.array([vmin - eps, vmin, vmin + eps], dtype=float)
        density = np.array([0.0, 1.0, 0.0], dtype=float)
        return support, density

    support = np.linspace(vmin, vmax, 100)
    try:
        kernel = stats.gaussian_kde(sample)
        density = kernel(support)
        return support, density
    except Exception:
        bins = int(np.sqrt(sample.size))
        bins = max(5, min(20, bins))
        counts, bin_edges = np.histogram(sample, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        if centers.size == 0:
            return None
        return centers, counts


def _extract_density_data(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        if 'density' in value and 'support' in value:
            try:
                density = np.asarray(value['density'], dtype=float).ravel()
                support = np.asarray(value['support'], dtype=float).ravel()
            except (TypeError, ValueError):
                return None
            if density.size == 0 or support.size == 0:
                return None
            if density.size != support.size:
                return None
            mask = np.isfinite(density) & np.isfinite(support)
            if not np.any(mask):
                return None
            density = density[mask]
            support = support[mask]
            order = np.argsort(support)
            return support[order], density[order]
        if 'values' in value:
            return _kde_from_values(np.asarray(value['values']))
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            first = np.asarray(value[0])
            second = np.asarray(value[1])
            if first.ndim == 1 and second.ndim == 1 and first.size == second.size:
                return first.astype(float), second.astype(float)
        return _kde_from_values(np.asarray(value))
    return None


class DensityPlot(StatisticPlot):
    """[PLOT] Density Plot."""

    name: str = "Density Plot"
    _description: str = textwrap.dedent("""\
        Density plots show smoothed distributions for numeric columns.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot renders kernel density estimates for numeric columns using
        precomputed density statistics when available.
        """)
    refs: list[dict] = []

    title: str = "Density plot"
    description: str = textwrap.dedent("""\
        The density plot displays kernel density estimates for numeric columns.
        """)
    group_by_feature: bool = True

    def __str__(self) -> str:
        return 'density'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        dataset: Dataset | None = None,
        **kwargs,
    ) -> 'DensityPlot':
        """Compute density plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        density_key = str(self)
        if density_key not in dataframe.index:
            _plot_placeholder("Density statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        if dataset is not None:
            numeric_columns = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
            columns_to_show = [col for col in dataframe.columns if col in numeric_columns]
        else:
            columns_to_show = list(dataframe.columns)

        density_row = dataframe.loc[density_key]
        entries: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        for col in columns_to_show:
            density_data = _extract_density_data(density_row.get(col))
            if density_data is not None:
                entries.append((col, density_data))

        if not entries:
            _plot_placeholder("No numeric density statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        n_plots = len(entries)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
        axes_list = np.atleast_1d(axes).ravel()

        for ax, (col, (support, density)) in zip(axes_list, entries):
            if support.size == 0 or density.size == 0 or support.size != density.size:
                ax.text(0.5, 0.5, "Invalid density data", ha='center', va='center')
                ax.axis('off')
                continue
            ax.plot(support, density, color='tab:blue')
            ax.fill_between(support, density, alpha=0.3, color='tab:blue')
            ax.set_title(_column_label(col, base_name))
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')

        for ax in axes_list[len(entries):]:
            ax.axis('off')

        if base_name:
            fig.suptitle(f"Density plot: {base_name}")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()

        plt.savefig(self._binary_image, format='png')
        return self
