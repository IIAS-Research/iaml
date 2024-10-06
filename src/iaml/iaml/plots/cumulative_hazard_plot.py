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
from ..plot import Plot, capture
from ..logger import Logger

class CumulativeHazardModelComparisonPlot(Plot):
    """
    [PLOT] Cumulative Hazard Model Comparison Plot using sksurv
    """

    @property
    def explain(self) -> str:
        """
        Return str explanation of the plot
        """
        return textwrap.dedent("""
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
    def compute(self, estimator, X:pd.DataFrame, y:pd.Series, 
                X_train:pd.DataFrame=None, y_train:pd.Series=None, 
                baseline_estimator=None, transform:bool=True,
                **kwargs) -> Plot:
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

        # Baseline model (CoxPH)
        try:
            if baseline_estimator is None:
                if X_train is not None and y_train is not None:
                    transform = True
                    baseline_estimator = CoxPHSurvivalAnalysis()
                    y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
                    baseline_estimator.fit(estimator.transform(X_train), y_train)

            if transform:
                pred_hazard_fn_cox = baseline_estimator.predict_cumulative_hazard_function(
                    estimator.transform(X)
                    )
            else:
                pred_hazard_fn_cox = baseline_estimator.predict_cumulative_hazard_function(X)

            mean_hazard_prob_cox = np.mean([fn.y for fn in pred_hazard_fn_cox], axis=0)
            mean_hazard_time_cox = pred_hazard_fn_cox[0].x

            model_name = baseline_estimator.name if hasattr(baseline_estimator, 'name') \
                else str(baseline_estimator)
            plt.step(mean_hazard_time_cox, mean_hazard_prob_cox, where="post", 
                    label=f"Baseline model ({model_name})", color="red", linestyle="--")
        except Exception:
            Logger().error(traceback.format_exc())

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
        plt.legend()

        plt.savefig(self._binary_image, format='png')
        plt.close()

        return self
    
    def suitable(self, type_of_target:str) -> bool:
        """
        Does this plot is usable for a given type_of_target?
        """
        return type_of_target in ['survival']
