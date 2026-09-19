"""[PLOT] Pair plot for descriptive statistics."""
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


def _sample_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if frame.shape[0] <= max_points:
        return frame
    indices = np.linspace(0, frame.shape[0] - 1, max_points, dtype=int)
    return frame.iloc[indices]


def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors='coerce')


def _extract_row_values(row: pd.Series) -> pd.DataFrame | None:
    if not isinstance(row, pd.Series):
        return None
    data: dict[str, np.ndarray] = {}
    min_len: int | None = None
    for col, value in row.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        arr: np.ndarray | None = None
        if isinstance(value, dict) and 'values' in value:
            arr = np.asarray(value['values'])
        elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(value)
        if arr is None:
            continue
        arr = arr.ravel()
        if arr.size == 0:
            continue
        if min_len is None or arr.size < min_len:
            min_len = int(arr.size)
        data[str(col)] = arr

    if not data or min_len is None or min_len < 2:
        return None

    trimmed = {col: values[:min_len] for col, values in data.items()}
    return pd.DataFrame(trimmed)


def _frame_from_stats(dataframe: pd.DataFrame, key: str) -> pd.DataFrame | None:
    if dataframe.empty or key not in dataframe.index:
        return None
    row = dataframe.loc[key]
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        row = row.iloc[0]
    return _extract_row_values(row)


def _looks_like_raw_data(dataframe: pd.DataFrame) -> bool:
    if dataframe.shape[0] < 2 or dataframe.shape[1] < 2:
        return False
    stats_keys = {
        'mean', 'median', 'min', 'max', 'std', 'variance', 'count', 'mode',
        'null_count', 'missing_rate', 'value_counts', 'range', 'iqr',
    }
    if any(str(label) in stats_keys for label in dataframe.index):
        return False
    if dataframe.index.dtype == object and dataframe.index.nunique() <= 2:
        return False
    numeric = _to_numeric_frame(dataframe)
    return np.isfinite(numeric.to_numpy()).any()


def _select_numeric_frame(
    dataframe: pd.DataFrame,
    dataset: Dataset | None,
    max_features: int,
    max_points: int,
) -> pd.DataFrame | None:
    if dataset is not None:
        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not numeric_columns:
            return None
        columns = numeric_columns[:max_features]
        frame = dataset.X[columns]
        frame = _to_numeric_frame(frame)
        frame = _sample_frame(frame, max_points)
        return frame

    if _looks_like_raw_data(dataframe):
        frame = _to_numeric_frame(dataframe)
        if frame.shape[1] > max_features:
            frame = frame.iloc[:, :max_features]
        frame = _sample_frame(frame, max_points)
        return frame

    return None


def _figure_size(n_features: int) -> tuple[float, float]:
    size = float(min(12.0, max(4.0, 2.2 * n_features)))
    return size, size


class PairPlot(StatisticPlot):
    """[PLOT] Pair Plot."""

    name: str = "Pair Plot"
    _description: str = textwrap.dedent("""\
        Pair plots show pairwise scatter plots between numeric features.
        """)
    _description_long: str = textwrap.dedent("""\
        This plot renders a scatter matrix for a small number of numeric features,
        highlighting pairwise relationships and marginal distributions.
        """)
    refs: list[dict] = []

    title: str = "Pair plot"
    description: str = textwrap.dedent("""\
        The pair plot shows pairwise scatter plots for small numeric feature sets.
        """)
    group_by_feature: bool = False

    def __str__(self) -> str:
        return 'pairplot'

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        dataset: Dataset | None = None,
        base_name: str | None = None,
        **kwargs,
    ) -> 'PairPlot':
        """Compute pair plot statistics."""
        self._binary_image = io.BytesIO()

        max_features = int(kwargs.get('max_features', 6))
        max_points = int(kwargs.get('max_points', 800))
        max_features = max(2, max_features)
        max_points = max(50, max_points)

        frame = _frame_from_stats(dataframe, str(self))
        if frame is None:
            for candidate in ('pair_plot', 'scatter_matrix', 'scatter'):
                frame = _frame_from_stats(dataframe, candidate)
                if frame is not None:
                    break

        if frame is None:
            frame = _select_numeric_frame(dataframe, dataset, max_features, max_points)

        if frame is None or frame.empty:
            _plot_placeholder("Pair plot data not available")
            plt.savefig(self._binary_image, format='png')
            return self

        frame = _to_numeric_frame(frame)
        frame = frame.dropna(how='all')
        frame = frame.loc[:, frame.notna().any(axis=0)]
        if frame.shape[1] < 2 or frame.shape[0] < 2:
            _plot_placeholder("Not enough numeric data for pair plot")
            plt.savefig(self._binary_image, format='png')
            return self

        if frame.shape[1] > max_features:
            frame = frame.iloc[:, :max_features]

        columns = [str(col) for col in frame.columns]
        n_features = len(columns)
        fig_w, fig_h = _figure_size(n_features)
        fig, axes = plt.subplots(n_features, n_features, figsize=(fig_w, fig_h))

        values = frame.to_numpy(dtype=float)
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
                if i == j:
                    data = values[:, i]
                    data = data[np.isfinite(data)]
                    if data.size == 0:
                        ax.text(0.5, 0.5, "No data", ha='center', va='center')
                        ax.axis('off')
                    else:
                        bins = int(np.sqrt(data.size))
                        bins = max(5, min(15, bins))
                        ax.hist(data, bins=bins, color='tab:blue', alpha=0.7)
                else:
                    x = values[:, j]
                    y = values[:, i]
                    mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(mask):
                        ax.text(0.5, 0.5, "No data", ha='center', va='center')
                        ax.axis('off')
                    else:
                        ax.scatter(x[mask], y[mask], s=10, alpha=0.6, color='tab:blue')

                if i == n_features - 1:
                    ax.set_xlabel(columns[j], rotation=45, ha='right')
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(columns[i])
                else:
                    ax.set_yticklabels([])

        title = "Pair plot"
        if base_name:
            title = f"Pair plot: {base_name}"
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(self._binary_image, format='png')
        return self
