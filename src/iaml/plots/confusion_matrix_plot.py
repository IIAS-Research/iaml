"""[PLOT] Confusion Matrix Plot"""
import textwrap

from yellowbrick.classifier import ConfusionMatrix

from ..metric_plot import MetricPlot, yellowbrick_plot


@yellowbrick_plot(ConfusionMatrix)
class ConfusionMatrixPlot(MetricPlot):
    """[PLOT] Confusion Matrix Plot"""

    title: str = "Confusion Matrix"
    description: str = textwrap.dedent("""
        The Confusion Matrix is a powerful visualization tool used to assess how well a classification model 
        is performing, particularly in identifying different medical conditions or diagnostic categories. 
        It provides a clear breakdown of true positive, false positive, true negative, and false negative rates.

        For example, in a healthcare setting, if a model is trained to classify whether a patient has 'diabetes', 
        'hypertension', or is 'healthy', the Confusion Matrix will show how many times the model made correct predictions 
        (true positives and true negatives) and where it made mistakes (false positives and false negatives).

        Each row in the matrix represents the actual condition of the patient, and each column represents the predicted 
        condition. This makes it easy to spot patterns in the model's predictions, such as whether it tends to misclassify 
        one condition as another.

        Doctors and data scientists rely on this plot to understand not only how often the model is right but also the 
        types of errors it makes. This information is crucial in healthcare, where reducing misdiagnoses can significantly 
        improve patient outcomes.
        """)

    @classmethod
    def suitable(cls, type_of_target: str) -> bool:
        return type_of_target in ['binary', 'multiclass', 'multilabel-indicator']
