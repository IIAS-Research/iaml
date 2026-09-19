"""[PLOT] Missingness heatmap for descriptive statistics."""
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


def _missing_row_from_stats(dataframe: pd.DataFrame) -> pd.Series | None:
    if dataframe.empty:
        return None
    if 'missing_rate' in dataframe.index:
        row = dataframe.loc['missing_rate']
    elif dataframe.shape[0] == 1:
        row = dataframe.iloc[0]
    else:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        row = row.iloc[0]
    return row


def _split_missing_column(column: Any) -> tuple[str, str]:
    column_str = str(column)
    if column_str.endswith('_all'):
        return column_str[:-4], 'all'
    if '_' in column_str:
        base, label = column_str.rsplit('_', 1)
        return base, label
    return column_str, 'all'


def _missing_frame_from_stats(
    dataframe: pd.DataFrame,
    dataset: Dataset | None,
) -> pd.DataFrame | None:
    row = _missing_row_from_stats(dataframe)
    if row is None:
        return None

    if dataset is not None and dataset.type_of_target != 'survival':
        columns = [
            col for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]
        if not columns:
            return None
        if dataset.type_of_target == 'continuous':
            values = []
            for col in columns:
                values.append(row.get(col, np.nan))
            return pd.DataFrame([values], index=['all'], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        labels = ['all'] + [str(label) for label in class_labels]
        data = np.full((len(labels), len(columns)), np.nan, dtype=float)
        for col_idx, col in enumerate(columns):
            for row_idx, label in enumerate(labels):
                key = f"{col}_{label}"
                data[row_idx, col_idx] = row.get(key, np.nan)
        return pd.DataFrame(data, index=labels, columns=columns)

    columns: list[str] = []
    labels: list[str] = []
    values: dict[tuple[str, str], float] = {}
    for col_name, value in row.items():
        base, label = _split_missing_column(col_name)
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


def _missing_frame_from_dataset(dataset: Dataset) -> pd.DataFrame | None:
    if dataset.type_of_target == 'survival':
        return None

    columns = [
        col for col in dataset.X.columns
        if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
    ]
    if not columns:
        return None

    frame = dataset.X[columns]
    overall = frame.isna().mean(axis=0)

    if dataset.type_of_target == 'continuous':
        return pd.DataFrame([overall.to_numpy()], index=['all'], columns=columns)

    grouped = frame.isna().groupby(dataset.y, sort=False).mean()
    labels = list(pd.unique(dataset.y))
    grouped = grouped.reindex(labels)
    grouped.index = [str(label) for label in grouped.index]

    overall_frame = pd.DataFrame([overall.to_numpy()], index=['all'], columns=columns)
    return pd.concat([overall_frame, grouped])


def _figure_size(n_cols: int, n_rows: int) -> tuple[float, float]:
    width = float(min(14.0, max(5.0, 0.6 * n_cols + 2.5)))
    height = float(min(10.0, max(3.5, 0.5 * n_rows + 2.0)))
    return width, height


class MissingnessHeatmapPlot(StatisticPlot):
    """[PLOT] Missingness Heatmap Plot."""

    name: str = "Missingness Heatmap"
    _description: str = textwrap.dedent("""\
        Missingness heatmaps summarize the percentage of missing values.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot displays a heatmap of missing rates per feature, optionally
        broken down by class labels for classification datasets.
        """)
    refs: list[dict] = []

    title: str = "Missingness heatmap"
    description: str = textwrap.dedent("""\
        The missingness heatmap shows missing value rates per feature.
        """)
    group_by_feature: bool = False

    def __str__(self) -> str:
        return 'missing_rate'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        dataset: Dataset | None = None,
        base_name: str | None = None,
        **kwargs,
    ) -> 'MissingnessHeatmapPlot':
        """Compute missingness heatmap statistics."""
        self._binary_image = io.BytesIO()

        missing_df = _missing_frame_from_stats(dataframe, dataset)
        if missing_df is None and dataset is not None:
            missing_df = _missing_frame_from_dataset(dataset)

        if missing_df is None or missing_df.empty:
            _plot_placeholder("Missingness statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        values = missing_df.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        n_rows, n_cols = masked.shape
        fig_size = _figure_size(n_cols, n_rows)
        fig, ax = plt.subplots(figsize=fig_size)

        image = ax.imshow(masked, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Missing rate')

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([str(label) for label in missing_df.columns], rotation=45, ha='right')
        ax.set_yticklabels([str(label) for label in missing_df.index])

        title = "Missingness heatmap"
        if base_name:
            title = f"Missingness heatmap: {base_name}"
        ax.set_title(title)
        ax.set_xlabel('Features')
        ax.set_ylabel('Group')

        tick_size = 10 if n_cols <= 12 else 8 if n_cols <= 20 else 6
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        plt.tight_layout()
        plt.savefig(self._binary_image, format='png')
        return self
