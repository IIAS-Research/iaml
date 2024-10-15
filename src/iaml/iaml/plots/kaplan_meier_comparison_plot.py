"""
[PLOT] Kaplan-Meier Model Comparison Survival Plot using sksurv
"""
import textwrap
import io
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from ..plot import MetricPlot, capture
from ..logger import Logger
from ..dataset import Dataset

class KaplanMeierModelComparisonPlot(MetricPlot):
    """
    [PLOT] Kaplan-Meier Model Comparison Survival
    """
    
    title = "Kaplan-Meier Model Comparison"
    description = textwrap.dedent("""
        The Kaplan-Meier Model Comparison Plot is a diagnostic tool used to evaluate the performance of 
        survival models by comparing predicted survival curves against the observed survival data.

        This plot is particularly useful for assessing how well a model can predict the time-to-event 
        outcome, such as time until death, disease recurrence, or failure. The observed Kaplan-Meier 
        survival curve represents the true survival probability over time, while the model's predicted 
        survival curves show the model's estimations.

        The x-axis represents time, and the y-axis represents the survival probability. Ideally, 
        the model-predicted survival curves should closely align with the observed Kaplan-Meier 
        curve, indicating good model performance. Discrepancies between the two curves highlight 
        areas where the model's predictions diverge from reality, signaling potential issues with 
        the model's predictive ability.

        Additionally, if possible, a Cox proportional hazards model is also trained to serve as 
        a baseline. This allows for a better understanding of model performances, as the Cox model 
        is a widely-used, interpretable model in survival analysis. By comparing more complex models 
        to this baseline, it becomes easier to gauge the improvement (or lack thereof) in predictive 
        accuracy.
        """)
    
    @capture
    def _compute(self, estimator, X:pd.DataFrame, y:pd.Series, 
                X_train:pd.DataFrame=None, y_train:pd.Series=None,
                transform:bool=True,
                **kwargs) -> MetricPlot:
        """
        Compute Kaplan-Meier survival plot with model predictions for comparison using sksurv.
        
        Parameters:
        - model: The survival model used to make predictions (e.g., CoxPH from sksurv)
        - X: The input data used for making predictions
        - durations: Series of observed survival times
        - event_observed: Series indicating whether the event occurred (1) or was censored (0)
        - groups: Optional Series indicating different groups for stratified survival analysis
        """
        self._binary_image = io.BytesIO()
        
        X_train, y_train = Dataset.fix_survival(X_train, y_train)
        X, y = Dataset.fix_survival(X, y)

        # Fit the Kaplan-Meier model on observed data
        event, time = zip(*y)
        
        # Observed data
        time, survival_prob = kaplan_meier_estimator(event, time)
        plt.step(time, survival_prob, where="post", label="Observed", color='blue')
        # Observed data
        # time, survival_prob = kaplan_meier_estimator(*zip(*y_train))
        # plt.step(time, survival_prob, where="post", label="Observed Train", color='blue', linestyle="--")
        
        
        # Current model
        survival_predictions = estimator.predict_survival_function(X)

        mean_survival_prob = np.mean([fn.y for fn in survival_predictions], axis=0)
        mean_survival_time = survival_predictions[0].x 

        plt.step(mean_survival_time, mean_survival_prob,
            where="post", label="Model prediction", color="green")

        # Customize and save the plot
        plt.title("Kaplan-Meier Curve vs Model Predicted Survival")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
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
