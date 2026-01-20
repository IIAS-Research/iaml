"""[PLOT] Bar plot for descriptive statistics."""
from __future__ import annotations

import io
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..plot import StatisticPlot, capture


def _is_missing(value: object) -> bool:
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
    return column_str.rsplit('_', 1)[-1] if '_' in column_str else 'all'


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


class BarPlot(StatisticPlot):
    """[PLOT] Bar Plot."""

    title: str = "Bar plot"
    description: str = textwrap.dedent("""\
        The bar plot shows descriptive statistics for categorical or numerical columns.
        Categorical plots show value counts per class, while numerical plots show metrics
        such as mean or variance per class.
        """)
    group_by_feature: bool = True

    @capture
    def compute(
        self,
        dataframe: pd.DataFrame,
        base_name: str | None = None,
        **kwargs) -> 'BarPlot':
        """Compute bar plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        base_name = base_name or _infer_base_name(list(dataframe.columns))

        if 'value_counts' in dataframe.index and dataframe.loc['value_counts'].notna().any():
            categories = []
            seen = set()
            for col in dataframe.columns:
                values = dataframe.at['value_counts', col]
                if _is_missing(values):
                    continue
                for cat, _ in values:
                    if cat not in seen:
                        seen.add(cat)
                        categories.append(cat)

            if not categories:
                _plot_placeholder("No categorical statistics available")
                plt.savefig(self._binary_image, format='png')
                return self

            labels = [_column_label(col, base_name) for col in dataframe.columns]
            data = {cat: [] for cat in categories}
            for col in dataframe.columns:
                values = dataframe.at['value_counts', col]
                value_dict = dict(values) if not _is_missing(values) else {}
                for cat in categories:
                    data[cat].append(value_dict.get(cat, 0))
            df = pd.DataFrame(data, index=labels)

            plt.figure()
            x = np.arange(len(df.index))
            width = min(0.8 / max(len(df.columns), 1), 0.2)
            for i, category in enumerate(df.columns):
                offset = (i - (len(df.columns) - 1) / 2) * width
                plt.bar(x + offset, df[category], width, label=category)
            plt.xticks(x, labels)
            plt.legend()
            plt.ylabel('Count')
            plt.title(f'[CAT] {base_name} Statistics')
            plt.tight_layout()
        elif 'mean' in dataframe.index and dataframe.loc['mean'].notna().any():
            numeric_df = dataframe.copy()
            for row in ['mode', 'value_counts', 'null_count', 'count']:
                if row in numeric_df.index:
                    numeric_df = numeric_df.drop(index=row)
            numeric_df = numeric_df.dropna(how='all')
            if numeric_df.empty:
                _plot_placeholder("No numeric statistics available")
                plt.savefig(self._binary_image, format='png')
                return self

            labels = [_column_label(col, base_name) for col in numeric_df.columns]
            x = np.arange(len(numeric_df.index))
            width = min(0.8 / max(len(numeric_df.columns), 1), 0.25)
            plt.figure()
            for i, col in enumerate(numeric_df.columns):
                offset = (i - (len(numeric_df.columns) - 1) / 2) * width
                plt.bar(x + offset, numeric_df[col], width, label=labels[i])
            plt.xticks(x, numeric_df.index, rotation=45, ha='right')
            plt.legend()
            plt.ylabel('Value')
            plt.title(f'[NUM] {base_name} Statistics')
            plt.tight_layout()
        else:
            _plot_placeholder("No statistics available for bar plot")

        plt.savefig(self._binary_image, format='png')
        return self
