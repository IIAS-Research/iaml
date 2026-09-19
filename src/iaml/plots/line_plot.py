"""[PLOT] Line plot for descriptive statistics."""
from __future__ import annotations

import io
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

from ..plot import StatisticPlot, capture


def _plot_placeholder(message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')


class LinePlot(StatisticPlot):
    """[PLOT] Line Plot."""

    title: str = "Line plot"
    description: str = textwrap.dedent("""\
        The line plot compares the null count and count statistics across columns.
        """)
    group_by_feature: bool = False

    @capture
    def compute(self, dataframe: pd.DataFrame, **kwargs) -> 'LinePlot':
        """Compute line plot statistics."""
        self._binary_image = io.BytesIO()

        if dataframe.empty:
            _plot_placeholder("No statistics available")
            plt.savefig(self._binary_image, format='png')
            return self

        if 'count' not in dataframe.index or 'null_count' not in dataframe.index:
            _plot_placeholder("Count statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        columns_to_show = [
            col for col in dataframe.columns
            if isinstance(col, str) and col.endswith('_all')
        ]
        if columns_to_show:
            labels = [col[:-4] for col in columns_to_show]
        else:
            columns_to_show = list(dataframe.columns)
            labels = [str(col) for col in columns_to_show]

        counts = dataframe.loc['count', columns_to_show]
        null_counts = dataframe.loc['null_count', columns_to_show]
        if counts.empty or null_counts.empty:
            _plot_placeholder("Count statistics not available")
            plt.savefig(self._binary_image, format='png')
            return self

        plt.figure()
        plt.plot(labels, null_counts.values, marker='o', label='Null Count', color='red')
        plt.plot(labels, counts.values, marker='s', label='Count', color='blue')
        plt.title('Null Count vs Count')
        plt.xlabel('Columns')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(self._binary_image, format='png')
        return self
