"""[PLOT] Class Prediction Error Plot"""
import textwrap

from yellowbrick.classifier import ClassPredictionError

from ..metric_plot import MetricPlot, yellowbrick_plot


@yellowbrick_plot(ClassPredictionError)
class ClassPredictionErrorPlot(MetricPlot):
    """[PLOT] Class Prediction Error Plot"""

    title: str = "Prediction Error Plot"
    description: str = textwrap.dedent("""
        The Class Prediction Error is a visualization that helps understand how well a machine 
        learning model is performing in predicting medical conditions or diagnoses. It shows 
        both the correct predictions made by the model and the mistakes it makes for each condition.

        For example, imagine you have a model that’s trained to identify different diseases 
        from patient data, such as predicting whether someone has diabetes, hypertension, 
        or is healthy. The Class Prediction Error plot would show, for each of these conditions, 
        how many times the model correctly identified the disease and how many times it made a 
        wrong prediction.

        For instance, if the model predicts "diabetes" for a patient who actually has "hypertension," 
        the plot will highlight this error. Similarly, it will also show how often the model 
        correctly identifies "healthy" patients versus when it mistakenly predicts they have 
        a disease.

        This visualization is especially helpful for doctors and data scientists because it clearly 
        shows where the model is making errors, making it easier to improve its accuracy, which is 
        critical in healthcare where correct predictions can have a big impact on patient outcomes.
        """)

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass', 'multilabel-indicator']
