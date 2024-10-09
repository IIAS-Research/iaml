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
                baseline_estimator=None, transform:bool=True,
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

        # Fit the Kaplan-Meier model on observed data
        event, time = zip(*y)
        
        # Observed data
        time, survival_prob = kaplan_meier_estimator(event, time)
        plt.step(time, survival_prob, where="post", label="Observed", color='blue')
        
        # Baseline
        try:
            # COX
            if baseline_estimator is None:
                if X_train is not None and y_train is not None:
                    transform = True
                    baseline_estimator = CoxPHSurvivalAnalysis()
                    y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
                    
                    baseline_estimator.fit(estimator.transform(X_train), y_train)

            if transform:
                pred_surv_fn = baseline_estimator.predict_survival_function(estimator.transform(X))
            else:
                pred_surv_fn = baseline_estimator.predict_survival_function(X)

            mean_survival_prob = np.mean([fn.y for fn in pred_surv_fn], axis=0)
            mean_survival_time = pred_surv_fn[0].x 
            
            model_name = baseline_estimator.name if hasattr(baseline_estimator, 'name') \
                else str(baseline_estimator)
            plt.step(mean_survival_time, mean_survival_prob, where="post", 
                label=f"Baseline model ({model_name})", color="red", linestyle="--")
        except RuntimeError:
            Logger().error(traceback.format_exc())
                
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
