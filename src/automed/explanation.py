"""
Enables steps to "explain" their processings and prediction results.
"""
from typing import TYPE_CHECKING

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
            metrics: dict['Metric', float] = None) -> None:
        if processings is None:
            processings = []

        if metrics is None:
            metrics = {}

        self.step = step
        self.processings = processings
        self.metrics = metrics

    @property
    def description(self) -> str:
        """
        Formats the description of a step with its configuration.

        Returns:
            str: Formatted description.
        """
        conf = { k: v['value'] for k, v in self.step.current_configuration.items() }

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
    
    def to_markdown(self, processings_limit: int = 20) -> str:
        """
        Renders the explanation as Markdown text.

        Returns:
            str: Markdown text.
        """
        if len(self.processings) == 0 and len(self.metrics) == 0:
            return ''

        confs = '\n'.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'
            for k, v in self.step.current_configuration.items()
        ])

        processings = '\n'.join([ f' - {p}' for p in self.processings[:processings_limit] ])

        metrics = '\n'.join([
            f'| `{m}` | **{v:.4f}** | *{m.explain()}* |'
            for m, v in self.metrics.items()
        ])

        processings_left = len(self.processings) - processings_limit

        return f"""
## {self.step.name}
**{self.description}**

{f'''
### Configuration
| Name | Description | Value |
| ---- | ----------- | ----- |
{confs}
''' if len(confs) > 0 else ""}

{f'''
### Processings
{processings}
{f" - *and **{processings_left}** more processings...*" if processings_left > 0 else ""}
''' if len(processings) > 0 else ""}

{f'''
### Metrics
| Metric name | Computed value | Description |
| ----------- | -------------- | ----------- |
{metrics}
''' if len(metrics) > 0 else ""}
        """
