"""[PLOT] Outlier plot for descriptive statistics."""
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


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


def _outlier_row_from_stats(dataframe: pd.DataFrame, key: str) -> pd.Series | None:
    if dataframe.empty:
        return None
    if key in dataframe.index:
        row = dataframe.loc[key]
    elif dataframe.shape[0] == 1:
        row = dataframe.iloc[0]
    else:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        row = row.iloc[0]
    return row


def _split_outlier_column(column: Any) -> tuple[str, str]:
    column_str = str(column)
    if column_str.endswith('_all'):
        return column_str[:-4], 'all'
    if '_' in column_str:
        base, label = column_str.rsplit('_', 1)
        return base, label
    return column_str, 'all'


def _outlier_frame_from_stats(
    dataframe: pd.DataFrame,
    dataset: Dataset | None,
    key: str,
) -> pd.DataFrame | None:
    row = _outlier_row_from_stats(dataframe, key)
    if row is None:
        return None

    if dataset is not None and dataset.type_of_target != 'survival':
        numeric_columns = list(dataset.get_columns_names_by_type(DataType.NUMERIC))
        if not numeric_columns:
            return None
        if dataset.type_of_target == 'continuous':
            values = [row.get(col, np.nan) for col in numeric_columns]
            return pd.DataFrame([values], index=['all'], columns=numeric_columns)

        class_labels = list(pd.unique(dataset.y))
        labels = ['all'] + [str(label) for label in class_labels]
        data = np.full((len(labels), len(numeric_columns)), np.nan, dtype=float)
        for col_idx, col in enumerate(numeric_columns):
            for row_idx, label in enumerate(labels):
                key_name = f"{col}_{label}"
                data[row_idx, col_idx] = row.get(key_name, np.nan)
        return pd.DataFrame(data, index=labels, columns=numeric_columns)

    columns: list[str] = []
    labels: list[str] = []
    values: dict[tuple[str, str], float] = {}
    for col_name, value in row.items():
        base, label = _split_outlier_column(col_name)
        if base not in columns:
            columns.append(base)
        if label not in labels:
            labels.append(label)
        values[(label, base)] = value

    if not columns or not labels:
        return None

    if 'all' in labels:
        labels = ['all'] + [label for label in labels if label != 'all']

    data = np.full((len(labels), len(columns)), np.nan, dtype=float)
    for row_idx, label in enumerate(labels):
        for col_idx, base in enumerate(columns):
            data[row_idx, col_idx] = values.get((label, base), np.nan)
    return pd.DataFrame(data, index=labels, columns=columns)


def _figure_size(n_cols: int) -> tuple[float, float]:
    width = float(min(14.0, max(6.0, 0.7 * n_cols + 2.5)))
    height = 4.5 if n_cols <= 8 else 5.5
    return width, height


class OutlierPlot(StatisticPlot):
    """[PLOT] Outlier Plot."""

    name: str = "Outlier Plot"
    _description: str = textwrap.dedent("""\
        Outlier plots show counts of outliers per column.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot visualizes outlier counts based on the 1.5*IQR rule for each
        numeric column, optionally broken down by class labels.
        """)
    refs: list[dict] = []

    title: str = "Outlier plot"
    description: str = textwrap.dedent("""\
        The outlier plot shows outlier counts per column using box/strip visuals.
        """)
    group_by_feature: bool = False

    def __str__(self) -> str:
        return 'outlier_count_iqr'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        dataset: Dataset | None = None,
        base_name: str | None = None,
        **kwargs,
    ) -> 'OutlierPlot':
        """Compute outlier plot statistics."""
        self._binary_image = io.BytesIO()

        outlier_key = str(self)
        outlier_df = _outlier_frame_from_stats(dataframe, dataset, outlier_key)
        if outlier_df is None or outlier_df.empty:
            _plot_placeholder("Outlier statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        columns = [col for col in outlier_df.columns]
        if not columns:
            _plot_placeholder("Outlier statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        values_per_column: list[np.ndarray] = []
        columns_to_plot: list[str] = []
        for col in columns:
            values = pd.to_numeric(outlier_df[col], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            columns_to_plot.append(col)
            values_per_column.append(values)

        if not columns_to_plot:
            _plot_placeholder("No numeric outlier statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        fig, ax = plt.subplots(figsize=_figure_size(len(columns_to_plot)))
        positions = np.arange(1, len(columns_to_plot) + 1)
        ax.boxplot(
            values_per_column,
            positions=positions,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={'facecolor': '#d9d9d9', 'edgecolor': '#555555'},
            medianprops={'color': '#333333'},
        )

        labels = list(outlier_df.index)
        n_labels = len(labels)
        offsets = np.linspace(-0.18, 0.18, n_labels) if n_labels > 1 else np.array([0.0])
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, max(1, n_labels)))

        for label_idx, label in enumerate(labels):
            row_values = pd.to_numeric(
                outlier_df.loc[label, columns_to_plot],
                errors='coerce',
            ).to_numpy(dtype=float)
            mask = np.isfinite(row_values)
            if not np.any(mask):
                continue
            x_positions = positions[mask] + offsets[label_idx]
            y_values = row_values[mask]
            ax.scatter(
                x_positions,
                y_values,
                s=28,
                color=colors[label_idx],
                alpha=0.85,
                label=str(label),
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([str(col) for col in columns_to_plot], rotation=30, ha='right')
        ax.set_ylabel('Outlier count')
        title = "Outlier count (IQR)"
        if base_name:
            title = f"Outlier count (IQR): {base_name}"
        ax.set_title(title)

        if n_labels > 1:
            ax.legend(title='Group', fontsize=8)

        plt.tight_layout()
        plt.savefig(self._binary_image, format='png')
        return self
