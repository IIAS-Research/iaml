"""
Enables steps to "explain" their processings and prediction results.
"""
from typing import TYPE_CHECKING
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import shap

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step


class Explanation:
    """
    Enables steps to "explain" their processings and prediction results.
    """
    def __init__(
            self,
            step: 'Step',
            processings: list[str] = None,
            metrics: dict['Metric', float] = None,
            shap_values: list = None) -> None:
        if processings is None:
            processings = []

        if metrics is None:
            metrics = {}

        self.step = step
        self.processings = processings
        self.metrics = metrics
        self.shap_values = shap_values

    @property
    def description(self) -> str:
        """
        Formats the description of a step with its configuration.

        Returns:
            str: Formatted description.
        """
        conf = { k: v['value'] for k, v in self.step.configuration.items() }

        return self.step.description.format(**conf)
    
    def add_processing(self, processing: str) -> None:
        """
        Adds a processing description to the list of processings.

        Args:
            processing (str): Processing description.
        """
        self.processings.append(processing)

    def set_metric(self, metric: 'Metric', value: float) -> None:
        """
        Adds a metric description to the list of metrics.

        Args:
            metric (Metric): Metric.
            value (float): Metric value.
        """
        self.metrics[metric] = value
    
    def to_plot(
            self,
            plot: str,
            ps: slice = None,
            scatter_features: list[str] = None) -> None:
        """
        Plot SHAP values.

        Args:
            plot (str): Plot to generate (one of "force", "scatter",
                "beeswarm", "heatmap", "bar"). Refer to the SHAP
                documentation for more details on these plots.
            ps (slice, optional): Specify the indexes of the SHAP values
                to plot. Defaults to all predictions. If plot is
                "force" or "waterfall", it will pick the first
                prediction matching the provided slice (defaults to
                first of all).
            scatter_features (list[str], optional): Specify the
                features to plot in the scatter plot (defaults to all).
        """
        if self.shap_values is None:
            raise RuntimeError('Cannot generate plots for this explanation without SHAP values.')

        if ps is None:
            ps = slice(0, len(self.shap_values))

        match plot:
            case 'force':
                shap.plots.force(self.shap_values[ps.start], show=False, matplotlib=True)
            case 'waterfall':
                shap.plots.waterfall(self.shap_values[ps.start], show=False)
            case 'scatter':
                if scatter_features is None:
                    scatter_features = self.shap_values.feature_names
                    print(scatter_features)

                shap.plots.scatter(self.shap_values[ps, scatter_features], show=False)
            case 'beeswarm':
                shap.plots.beeswarm(self.shap_values[ps], show=False)
            case 'heatmap':
                shap.plots.heatmap(self.shap_values[ps], show=False)
            case 'bar':
                shap.plots.bar(self.shap_values[ps], show=False)
            case _:
                raise RuntimeError(f'Unknown plot type ({plot}).')
    
    def to_markdown_data_uri_plot(self, plot: str, ps: slice = None, **kw):
        """
        Plot SHAP values and encode the plot image into a Markdown
        image.

        Args:
            plot (str): Plot to generate (one of "force", "scatter",
                "beeswarm", "heatmap", "bar"). Refer to the SHAP
                documentation for more details on these plots.
            ps (slice, optional): Specify the indexes of the SHAP values
                to plot. Defaults to all predictions. If plot is
                "force", it will pick the first prediction matching
                the provided slice (defaults to first of all).
            **kw (dict): Parameters to pass to matplotlib.
        
        Returns:
            str: Base64-encoded plot image.
        """
        self.to_plot(plot, ps, scatter_features=['Age'])

        buffer = io.BytesIO()
        plt.savefig(buffer, bbox_inches='tight', **kw)
        buffer.seek(0)

        plt.close()

        b64 = base64.b64encode(buffer.read()).decode()

        return f'![{plot} plot](data:image/png;base64,{b64})'
    
    def to_markdown_conf(self) -> str:
        """
        Renders the step configurations for this explanation as
        Markdown text.

        Returns
            str: Markdown text.
        """
        confs = '\n'.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'
            for k, v in self.step.configuration.items()
        ])

        return f"""
### Configuration
| Name | Description | Value |
| ---- | ----------- | ----- |
{confs}
        """ if len(confs) > 0 else ""
    
    def to_markdown_processings(self, processings_limit: int = 20) -> str:
        """
        Renders the processings for this explanation as Markdown text.
        """
        processings = '\n'.join([ f' - {p}' for p in self.processings[:processings_limit] ])
        processings_left = len(self.processings) - processings_limit

        return f"""
### Processings
{processings}
{f" - *and **{processings_left}** more processings...*" if processings_left > 0 else ""}
        """ if len(processings) > 0 else ""
    
    def to_markdown_metrics(self) -> str:
        """
        Renders model metrics for this explanation as Markdown text.

        Returns
            str: Markdown text.
        """
        metrics = '\n'.join([
            f'| `{m}` | **{v:.4f}** | *{m.explain()}* |'
            for m, v in self.metrics.items()
        ])

        return f"""
### Metrics
| Metric name | Computed value | Description |
| ----------- | -------------- | ----------- |
{metrics}
""" if len(metrics) > 0 else ""
    
    def to_markdown_shap(self) -> str:
        """
        Renders SHAP values for this explanation as Markdown text.

        Returns
            str: Markdown text.
        """
        if self.shap_values is not None:
            feature_names = self.shap_values.feature_names
            feature_values = np.abs(self.shap_values.values).mean(0)
            feature_impacts = '\n'.join(map(
                lambda i: f'| `{i[0]}` | **{i[1]:.3f}** |',
                list(zip(feature_names, feature_values))))
        
        return f"""
### Features impact
| Feature | Mean impact (SHAP value) |
| ------- | ------------------------ |
{feature_impacts}
""" if self.shap_values is not None else ""

    def to_markdown_plots(self) -> str:
        """
        Generates SHAP plots for this explanation, and renders them as
        Markdown text.

        Returns
            str: Markdown text.
        """
        if self.shap_values is not None:
            features = self.shap_values.feature_names

            values = self.shap_values[0].values
            force_shap, force_feature = max(zip(values, features), key=lambda v: abs(v[0]))

            mean_shap = np.abs(self.shap_values.values).mean(axis=0)
            bar_shap, bar_feature = max(zip(mean_shap, features), key=lambda v: v[0])

        return f"""
### SHAP plots
#### Force plot
{self.to_markdown_data_uri_plot('force')}

***Reading**: For this prediction, `{force_feature}` impacts the final prediction value by **{force_shap:.3f}**.*

#### Waterfall plot
{self.to_markdown_data_uri_plot('waterfall')}

***Reading**: For this prediction, `{force_feature}` impacts the final prediction value by **{force_shap:.3f}**.*

#### Beeswarm plot
{self.to_markdown_data_uri_plot('beeswarm')}

#### Heatmap plot
{self.to_markdown_data_uri_plot('heatmap')}

#### Scatter plot
{self.to_markdown_data_uri_plot('scatter')}

#### Bar plot
{self.to_markdown_data_uri_plot('bar')}

***Reading**: `{bar_feature}` has an absolute impact of **{bar_shap:.3f}** on the average final prediction value.*
""" if self.shap_values is not None else ""
    
    def to_markdown(self, processings_limit: int = 20) -> str:
        """
        Renders the explanation as Markdown text.

        Returns:
            str: Markdown text.
        """
        if len(self.processings) == 0 and len(self.metrics) == 0 and self.shap_values is None:
            return ''

        return f"""
## {self.step.name}
**{self.description}**

{self.to_markdown_conf()}
{self.to_markdown_processings(processings_limit)}
{self.to_markdown_metrics()}
{self.to_markdown_shap()}
{self.to_markdown_plots()}
        """
