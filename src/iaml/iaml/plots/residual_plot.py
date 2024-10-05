"""
[PLOT] Residuals Plot
"""
import textwrap
import io
import pandas as pd
from yellowbrick.regressor import ResidualsPlot as ybResidualsPlot

from ..plot import Plot, capture

class ResidualsPlot(Plot):
    """
    [PLOT] Residuals Plot
    """
    
    @property
    def explain(self) -> str:
        """
        Return str explanation of the plot
        """
        return textwrap.dedent("""
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
    
    @capture
    def compute(self, estimator:'IAMLPipeline', 
            X:pd.DataFrame, y:pd.DataFrame, 
            X_train:pd.DataFrame=None, y_train:list=None, **kwargs) -> Plot:
        """
        Compute plot given X, y.
        """
        self._binary_image = io.BytesIO()
        
        self.__visualizer = ybResidualsPlot(estimator)
        self.__visualizer.fit(X_train, y_train,)
        self.__visualizer.score(X, y)
        self.__visualizer.poof(self._binary_image)
        
        return self
    
    def suitable(self, type_of_target:str) -> bool:  # pylint: disable=unused-argument
        """
        Does this plot is usable for a given type_of_target ?
        """
        return type_of_target == 'continuous'
