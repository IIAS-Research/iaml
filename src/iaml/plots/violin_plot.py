"""[PLOT] Violin plot for descriptive statistics."""
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


def _extract_violin_groups(
    value: Any,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]:
    if _is_missing(value) or not isinstance(value, dict):
        return []
    entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for category, stats in value.items():
        if not isinstance(stats, dict):
            continue
        density = stats.get('density')
        support = stats.get('support')
        if density is None or support is None:
            continue
        try:
            density_arr = np.asarray(density, dtype=float).ravel()
            support_arr = np.asarray(support, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if density_arr.size == 0 or support_arr.size == 0:
            continue
        if density_arr.size != support_arr.size:
            continue
        mask = np.isfinite(density_arr) & np.isfinite(support_arr)
        if not np.any(mask):
            continue
        density_arr = density_arr[mask]
        support_arr = support_arr[mask]
        order = np.argsort(support_arr)
        density_arr = density_arr[order]
        support_arr = support_arr[order]

        quartiles = None
        if 'quartiles' in stats:
            try:
                q_values = np.asarray(stats['quartiles'], dtype=float).ravel()
            except (TypeError, ValueError):
                q_values = np.asarray([], dtype=float)
            if q_values.size >= 3 and np.all(np.isfinite(q_values[:3])):
                quartiles = q_values[:3]
        entries.append((str(category), support_arr, density_arr, quartiles))
    return entries


def _render_violin(
    ax: plt.Axes,
    groups: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]],
) -> None:
    if not groups:
        ax.text(0.5, 0.5, "No violin data", ha='center', va='center')
        ax.axis('off')
        return
    max_density = max(float(np.nanmax(density)) for _, _, density, _ in groups)
    if not np.isfinite(max_density) or max_density <= 0:
        ax.text(0.5, 0.5, "Invalid density data", ha='center', va='center')
        ax.axis('off')
        return

    scale = 0.4 / max_density
    for idx, (category, support, density, quartiles) in enumerate(groups):
        width = density * scale
        ax.fill_betweenx(
            support,
            idx - width,
            idx + width,
            alpha=0.6,
            edgecolor='black',
            linewidth=0.8,
        )
        if quartiles is not None:
            q1, median, q3 = quartiles
            ax.plot([idx, idx], [q1, q3], color='black', linewidth=2)
            ax.plot([idx - 0.08, idx + 0.08], [median, median], color='black', linewidth=2)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for label, *_ in groups], rotation=30, ha='right')
    ax.set_xlabel('Category')
    ax.set_ylabel('Target')


class ViolinPlot(StatisticPlot):
    """[PLOT] Violin Plot."""

    name: str = "Violin Plot"
    _description: str = textwrap.dedent("""\
        Violin plots show target distributions per categorical value.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot renders violin distributions for categorical features using
        precomputed density statistics for a continuous target.
        """)
    refs: list[dict] = []

    title: str = "Violin plot"
    description: str = textwrap.dedent("""\
        The violin plot shows target distributions per categorical feature.
        """)
    group_by_feature: bool = True

    def __str__(self) -> str:
        return 'violin'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        dataset: Dataset | None = None,
        **kwargs,
    ) -> 'ViolinPlot':
        """Compute violin plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        violin_key = str(self)
        if violin_key not in dataframe.index:
            _plot_placeholder("Violin statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        if dataset is not None:
            categorical_columns = set(dataset.get_columns_names_by_type(DataType.CATEGORICAL))
            columns_to_show = [col for col in dataframe.columns if col in categorical_columns]
        else:
            columns_to_show = list(dataframe.columns)

        violin_row = dataframe.loc[violin_key]
        entries: list[tuple[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]]] = []
        for col in columns_to_show:
            groups = _extract_violin_groups(violin_row.get(col))
            if groups:
                entries.append((col, groups))

        if not entries:
            _plot_placeholder("No violin statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        n_plots = len(entries)
        n_cols = 1 if n_plots == 1 else 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes_list = np.atleast_1d(axes).ravel()

        for ax, (col, groups) in zip(axes_list, entries):
            _render_violin(ax, groups)
            ax.set_title(_column_label(col, base_name))

        for ax in axes_list[len(entries):]:
            ax.axis('off')

        if base_name:
            fig.suptitle(f"Violin plot: {base_name}")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()

        plt.savefig(self._binary_image, format='png')
        return self
