"""
[PLOT] Prediction Error Plot
"""
import textwrap
import io
import pandas as pd
from yellowbrick.regressor import PredictionError

from ..plot import MetricPlot, capture

class PredictionErrorPlot(MetricPlot):
    """
    [PLOT] Prediction Error Plot
    """
    
    title = "Prediction Error"
    description = textwrap.dedent("""
        The Prediction Error Plot is a diagnostic tool used to visualize the performance of regression models. 
        It helps assess how well a model's predictions align with the actual values in a continuous prediction 
        setting, such as predicting medical measurements like blood pressure, heart rate, or glucose levels.

        The plot shows the actual target values on the x-axis and the predicted values on the y-axis. A perfect 
        model would have all points lying on a 45-degree diagonal line, representing perfect predictions. Deviations 
        from this line indicate errors in the predictions.

        For instance, if you're building a model to predict a patient's blood pressure based on certain health metrics, 
        the Prediction Error Plot will show how closely the model's predictions match the actual measurements. If the 
        points are scattered away from the diagonal, it indicates that the model is making large prediction errors.

        This plot is useful for identifying both systematic errors (consistent overestimation or underestimation) and 
        random errors (scattering of points) in the model, which is crucial in healthcare settings where accurate 
        predictions can directly impact patient care.
        """)
    
    @capture
    def _compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y:pd.DataFrame, **kwargs) -> MetricPlot:
        """
        Compute plot given X, y.
        """
        self._binary_image = io.BytesIO()
        self.__visualizer = PredictionError(estimator, is_fitted=True)
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return type_of_target == 'continuous'
