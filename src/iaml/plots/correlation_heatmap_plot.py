"""[PLOT] Correlation heatmap for descriptive statistics."""
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


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors='coerce')
    return numeric


def _square_corr_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame.empty:
        return None
    if frame.shape[0] != frame.shape[1]:
        return None
    if set(frame.index) != set(frame.columns):
        return None
    ordered = frame.reindex(index=frame.index, columns=frame.index)
    numeric = _to_numeric_frame(ordered)
    if not np.isfinite(numeric.to_numpy()).any():
        return None
    return numeric


def _corr_from_value(value: Any) -> pd.DataFrame | None:
    if _is_missing(value):
        return None
    if isinstance(value, pd.DataFrame):
        return _square_corr_from_dataframe(value)
    if isinstance(value, dict):
        matrix = None
        if 'matrix' in value:
            matrix = value['matrix']
        elif 'values' in value:
            matrix = value['values']
        if matrix is None:
            return None
        df = pd.DataFrame(matrix)
        labels = value.get('labels')
        if labels is not None:
            df.index = labels
            df.columns = labels
        if 'index' in value:
            df.index = value['index']
        if 'columns' in value:
            df.columns = value['columns']
        return _square_corr_from_dataframe(df)
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim == 2:
            return _square_corr_from_dataframe(pd.DataFrame(array))
    return None


def _corr_from_row(row: pd.Series) -> pd.DataFrame | None:
    if not isinstance(row, pd.Series):
        return None
    for item in row:
        corr = _corr_from_value(item)
        if corr is not None:
            return corr
    return None


def _extract_corr_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame | None:
    corr_df = _square_corr_from_dataframe(dataframe)
    if corr_df is not None:
        return corr_df
    for key in ('correlation', 'correlation_matrix', 'corr', 'corr_matrix'):
        if key in dataframe.index:
            return _corr_from_row(dataframe.loc[key])
    return None


def _figure_size(n_features: int) -> float:
    if n_features <= 1:
        return 4.0
    return float(min(12.0, max(4.5, 0.5 * n_features + 2.0)))


class CorrelationHeatmapPlot(StatisticPlot):
    """[PLOT] Correlation Heatmap Plot."""

    name: str = "Correlation Heatmap"
    _description: str = textwrap.dedent("""\
        Correlation heatmaps summarize relationships between numeric features.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot displays a correlation matrix for numeric columns, helping
        identify strong positive or negative relationships.
        """)
    refs: list[dict] = []

    title: str = "Correlation heatmap"
    description: str = textwrap.dedent("""\
        The correlation heatmap visualizes pairwise correlations among numeric columns.
        """)
    group_by_feature: bool = False

    def __str__(self) -> str:
        return 'correlation'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        dataset: Dataset | None = None,
        base_name: str | None = None,
        **kwargs,
    ) -> 'CorrelationHeatmapPlot':
        """Compute correlation heatmap statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        corr_df = _extract_corr_dataframe(dataframe)

        if corr_df is None and dataset is not None:
            numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
            if numeric_columns:
                corr_df = dataset.X[numeric_columns].corr()

        if corr_df is None or corr_df.empty:
            _plot_placeholder("Correlation statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        corr_df = _to_numeric_frame(corr_df)
        if not np.isfinite(corr_df.to_numpy()).any():
            _plot_placeholder("Correlation statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        if dataset is not None:
            numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
            if numeric_columns:
                available = [col for col in corr_df.index if col in numeric_columns]
                if available:
                    corr_df = corr_df.loc[available, available]

        if corr_df.empty:
            _plot_placeholder("Correlation statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        n_features = corr_df.shape[0]
        fig_size = _figure_size(n_features)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        values = corr_df.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        labels = [str(label) for label in corr_df.columns]
        ticks = np.arange(n_features)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        label_size = 10 if n_features <= 12 else 8 if n_features <= 20 else 6
        ax.tick_params(axis='both', which='major', labelsize=label_size)

        title = "Correlation heatmap"
        if base_name:
            title = f"Correlation heatmap: {base_name}"
        ax.set_title(title)
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')
        plt.tight_layout()
        plt.savefig(self._binary_image, format='png')
        return self
