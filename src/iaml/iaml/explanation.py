"""Enables steps to "explain" their processings and prediction results."""
from typing import TYPE_CHECKING
import textwrap
import numpy as np
import shap
from .plots.shap_plot import ShapPlot

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step


class Explanation:
    """Enables steps to "explain" their processings and prediction results.
    
    :param Step step: The step to explain
    :param list[str], optional processings: Processing descriptions. Default to None.
    :param dict[Metric, float], optional metrics: Metrics to append to the explanation. 
        Default to None.
    :param shap.Explanation, optional shap_values: Shap values to append to the explanation.
        Default to None.
    """
    def __init__(
        self,
        step: 'Step',
        processings: list[str] = None,
        metrics: dict['Metric', float] = None,
        shap_values: shap.Explanation = None) -> None:
        if processings is None:
            processings = []

        if metrics is None:
            metrics = {}

        self.step = step
        """The step that'll be explained."""

        self.processings = processings
        """Processing descriptions."""

        self.metrics = metrics
        """Metrics dictionnary."""

        self.shap_values = shap_values
        """Shap values to append to the explanation."""

    @property
    def description(self) -> str:
        """Formats the description of a step with its configuration.

        :return: Formatted description.
        """
        conf = { k: v['value'] for k, v in self.step.configuration.items() }

        return self.step.description.format(**conf)

    def add_processing(self, processing: str) -> None:
        """Adds a processing description to the list of processings.

        :param str processings: Processing description.
        """
        self.processings.append(processing)

    def set_metric(self, metric: 'Metric', value: float) -> None:
        """Add a metric and it's value to the metrics dictionnary.

        :param Metric metric: Metric to add to the dictionnary of metrics.
        :param float value: Metric value
        """
        self.metrics[metric] = value

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

    def to_markdown_conf(self) -> str:
        """Renders the step configurations for this explanation as
        Markdown text.

       :return: Markdown document.
        """
        confs = '\n'.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'
            for k, v in self.step.configuration.items()
        ])

        return textwrap.dedent(f"""
            ### Configuration
            | Name | Description | Value |
            | ---- | ----------- | ----- |
            {confs}
            """) if len(confs) > 0 else ""

    def to_markdown_processings(self, processings_limit: int = 20) -> str:
        """Renders the processings for this explanation as Markdown text.

        :param int, optional processings_limit: Limit the number of processings that are displayed 
            in the Markdown document (defaults to 20).
        :return: Markdown document.
        """
        processings = '\n'.join([ f' - {p}' for p in self.processings[:processings_limit] ])
        processings_left = len(self.processings) - processings_limit

        return textwrap.dedent(f"""
            ### Processings
            {processings}
            {f" - *and **{processings_left}** more processings...*" if processings_left > 0 else ""}
            """) if len(processings) > 0 else ""

    def to_markdown_metrics(self) -> str:
        """Renders model metrics for this explanation as Markdown text.

        :return: Markdown document.
        """
        metrics = '\n'.join([
            f'| `{m}` | **{v:.4f}** | *{m.explain()}* |'
            for m, v in self.metrics.items()
        ])

        return textwrap.dedent(f"""
            ### Metrics
            | Metric name | Computed value | Description |
            | ----------- | -------------- | ----------- |
            {metrics}
            """) if len(metrics) > 0 else ""

    def to_markdown_shap(self) -> str:
        """Renders SHAP values for this explanation as Markdown text.

        :return: Markdown document.
        """
        feature_impacts = ""
        if self.shap_values is not None:
            feature_names = self.shap_values.feature_names
            feature_values = np.abs(self.shap_values.values).mean(axis=0)
            feature_impacts = '\n'.join(map(
                lambda i: f'| `{i[0]}` | **{np.mean(i[1]):.3f}** |',
                list(zip(feature_names, feature_values))))

        return textwrap.dedent(f"""
            ### Features impact
            | Feature | Mean impact (SHAP value) |
            | ------- | ------------------------ |
            {feature_impacts}
            """) if self.shap_values is not None else ""

    def features_impacts(self) -> list[dict[str, str]]:
        """Return features impacts

        :return: name and impact of each feature
        """
        if self.shap_values is not None:
            feature_names = self.shap_values.feature_names
            feature_values = np.abs(self.shap_values.values).mean(axis=0)
            return [{'name': key, 'value': value}
                for key, value in list(zip(feature_names, feature_values))]
        return []

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

        markdown = ['\n\n'.join(p.to_markdown() for p in self.to_plots(plots))]

        return textwrap.dedent(f"""
            ### SHAP plots

            {markdown}
            """)

    def to_markdown(
        self,
        processings_limit: int = 20,
        plots: list[str] = None) -> str:
        """Renders the explanation as Markdown text.

        :param int, optional processing_limit: Limit the number of
            processings that are displayed in the Markdown document
            (defaults to 20).
        :return: Markdown document
        """
        if len(self.processings) == 0 and len(self.metrics) == 0 and self.shap_values is None:
            return ''

        return textwrap.dedent(f"""
            ## {self.step.name}
            **{self.step.description}**

            {self.to_markdown_conf()}
            {self.to_markdown_processings(processings_limit)}
            {self.to_markdown_metrics()}
            {self.to_markdown_shap()}
            {self.to_markdown_plots(plots)}
            """)
