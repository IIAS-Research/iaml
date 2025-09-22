"""
[PLOT] Cumulative Hazard Model Comparison Plot using sksurv
"""
import io
import traceback
import textwrap
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..plot import MetricPlot, capture
from ..logger import Logger
from ..dataset import Dataset

class CumulativeHazardModelComparisonPlot(MetricPlot):
    """
    [PLOT] Cumulative Hazard Model Comparison Plot using sksurv
    """

    title = "Cumulative Hazard"
    description = textwrap.dedent("""
        The Cumulative Hazard Model Comparison Plot is a diagnostic tool used to evaluate the performance of 
        survival models by comparing predicted cumulative hazard functions against the observed cumulative hazards.

        The x-axis represents time, and the y-axis represents the cumulative hazard. Ideally, 
        the model-predicted cumulative hazard curves should closely align with the observed curves, 
        indicating good model performance. Discrepancies between the two curves highlight 
        areas where the model's predictions diverge from reality, signaling potential issues with 
        the model's predictive ability.

        Additionally, a Cox proportional hazards model can be trained to serve as a baseline for comparison.
        """)

    @capture
    def _compute(self, estimator, X:pd.DataFrame, y:pd.Series, 
                X_train:pd.DataFrame=None, y_train:pd.Series=None,
                transform:bool=True,
                **kwargs) -> MetricPlot:
        """
        Compute cumulative hazard plot with model predictions for comparison using sksurv.
        
        Parameters:
        - estimator: The survival model used to make predictions (e.g., CoxPH from sksurv)
        - X: The input data used for making predictions
        - y: Series of observed survival times and event indicators
        - X_train: Optional training data used to fit a baseline model
        - y_train: Optional survival times and event indicators for the training set
        """
        self._binary_image = io.BytesIO()

        # Fit the cumulative hazard model on observed data
        event, time = zip(*y)

        # Observed cumulative hazard using nelson_aalen_estimator
        time, cumulative_hazard = nelson_aalen_estimator(event, time)
        plt.step(time, cumulative_hazard, where="post", label="Observed", color='blue')

        # Current model prediction
        hazard_predictions = estimator.predict_cumulative_hazard_function(X)

        mean_hazard_prob = np.mean([fn.y for fn in hazard_predictions], axis=0)
        mean_hazard_time = hazard_predictions[0].x

        plt.step(mean_hazard_time, mean_hazard_prob,
                where="post", label="Model prediction", color="green")

        # Customize and save the plot
        plt.title("Cumulative Hazard Curve vs Model Predicted")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Hazard")
        plt.ylim([0, 1])
        plt.legend()

        plt.savefig(self._binary_image, format='png')
        plt.close()

        return self
    
    @classmethod
    def suitable(cls, type_of_target:str) -> bool:
        """
        Does this plot is usable for a given type_of_target?
        """
        return type_of_target in ['survival']
