"""Enables steps to "explain" their processings and prediction results."""
from __future__ import annotations

import textwrap

import numpy as np
import shap

from .plots.shap_plot import ShapPlot


class Explanation:
    """Enables steps to "explain" their processings and prediction results.

    :param shap.Explanation, optional shap_values: Shap values to append to the explanation.
        Default to None.
    """
    def __init__(
        self,
        shap_values: shap.Explanation = None) -> None:
        self.shap_values = shap_values
        """Shap values to append to the explanation."""

    def to_plot(
            self,
            plot: str,
            ps: slice = None,
            scatter_feature: list[str] = None) -> ShapPlot:
        """Plot SHAP values.

        :param str plot: Plot to generate (one of "force", "scatter",
            "beeswarm", "heatmap", "bar"). Refer to the SHAP
            documentation for more details on these plots.
        :param slice, optional ps: Specify the indexes of the SHAP values
            to plot. Defaults to all predictions. If plot is
            "force" or "waterfall", it will pick the first
            prediction matching the provided slice (defaults to
            first of all). Default to None.
        :param list[str], optional scatter_feature: Specify the feature to
            plot in the scatter plot (defaults to first). Default to None.
        :raise RuntimeError: Cannot generate plots for this explanation without SHAP values.
        :return: object containing shap plot
        """
        if self.shap_values is None:
            raise RuntimeError('Cannot generate plots for this explanation without SHAP values.')

        shap_plot = None
        if plot == 'scatter':
            shap_plot = ShapPlot(plot, self.shap_values, ps=ps, scatter_feature=scatter_feature)
        else:
            shap_plot = ShapPlot(plot, self.shap_values, ps=ps)

        return shap_plot

    def to_plots(self, plots: list[str] = None) -> list[ShapPlot]:
        """Generate several plots

        :param list[str], optional plots: list of plots to generate. If None are provided, it will
            generate ['force', 'waterfall', 'beeswarm', 'scatter', 'heatmap', 'bar'].
        :return: list of plots
        """
        if plots is None:
            plots = ['force', 'waterfall', 'beeswarm', 'scatter', 'heatmap', 'bar']
        return [self.to_plot(p) for p in plots]

    def features_importance(self) -> dict[str, str]:
        """Return features importance

        :return: name and impact of each feature
        """
        if self.shap_values is None:
            return []

        feature_names = self.shap_values.feature_names
        feature_values = np.abs(self.shap_values.values).mean(axis=0)

        return dict(zip(feature_names, feature_values))

    def to_markdown_shap(self) -> str:
        """Renders SHAP values for this explanation as Markdown text.

        :return: Markdown document.
        """
        if self.shap_values is None:
            return ''

        feature_importance = '\n            '.join([
            f'| `{name}` | **{np.mean(value):.3f}** |'
            for name, value in self.features_importance().items()
        ])

        return textwrap.dedent(f"""\
            ### Features impact
            | Feature | Mean impact (SHAP value) |
            | ------- | ------------------------ |
            {feature_importance}
            """)

    def to_markdown_plots(self, plots: list[str] = None) -> str:
        """Generates SHAP plots for this explanation, and renders them as
        Markdown text.

        :param list[str], optional plots: list of plots to generate
        :return: Markdown text.
        """
        if plots is None:
            plots = ['force', 'waterfall', 'beeswarm', 'scatter', 'heatmap', 'bar']

        if self.shap_values is None or len(plots) == 0:
            return ""

        return '\n'.join([ p.to_markdown() for p in self.to_plots(plots) ])
