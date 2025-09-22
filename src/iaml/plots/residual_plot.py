"""
[PLOT] Residuals Plot
"""
import textwrap

from yellowbrick.regressor import ResidualsPlot as ybResidualsPlot

from ..metric_plot import MetricPlot, yellowbrick_plot


@yellowbrick_plot(ybResidualsPlot)
class ResidualsPlot(MetricPlot):
    """[PLOT] Residuals Plot"""

    title: str = "Residuals Plot"
    description: str = textwrap.dedent("""
        The Residuals Plot is a diagnostic tool used to evaluate the performance of a regression model. 
        In the context of predicting continuous medical outcomes, such as blood pressure, cholesterol levels, 
        or other measurements, this plot helps assess how well the model's predictions match the actual 
        observed values.

        Residuals are the differences between the predicted values and the actual values. A well-performing 
        regression model should have residuals that are randomly scattered around zero. Patterns in the residuals 
        (such as curvature or clustering) can indicate that the model is not capturing certain relationships in 
        the data.

        For example, if you're building a model to predict a patient's cholesterol level based on various 
        health metrics, the Residuals Plot would show whether the model consistently overestimates or underestimates 
        values or if there are systematic errors.

        Doctors and data scientists use this plot to detect whether the model is biased in its predictions and 
        whether certain patterns or trends remain unexplained, which can be critical in refining models used 
        for predicting medical outcomes.
        """)

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:
        return type_of_target == 'continuous'
