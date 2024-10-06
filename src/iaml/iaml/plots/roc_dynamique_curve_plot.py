"""
[PLOT] ROC Dynamique Curve for Survival Models using sksurv
"""
import textwrap
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc

from ..plot import Plot, capture

class ROCDynamiqueCurvePlot(Plot):
    """
    [PLOT] ROC Dynamique Curve for Survival Models using sksurv
    """
    
    @property
    def explain(self) -> str:
        """
        Return str explanation of the plot
        """
        return textwrap.dedent("""
            This curve represents how well a predictive survival model is able to distinguish 
            between patients who experience an event (like death or a heart attack) at different 
            points in time and those who do not. The y-axis shows the AUC (Area Under the Curve), 
            which is a measure of how good the model is at making this distinction—the closer to 1, 
            the better the model performs. The x-axis represents time, showing different follow-up 
            periods after the initial observation.

            As time progresses, the curve helps us see if the model's predictions remain accurate 
            or start to decline. For example, in a medical study predicting patient survival after 
            a heart attack, this curve would indicate how well the model distinguishes between 
            patients who pass away versus those who survive, over several months or years. 
            A high AUC value means the model is very good at predicting outcomes, while a lower 
            value suggests it struggles to differentiate between high-risk and low-risk patients as 
            time goes on.""")
    
    @capture
    def compute(self, estimator, X: pd.DataFrame, y: pd.Series,
            y_train: pd.Series=None, **kwargs) -> Plot:
        """
        Compute AUC Dynamique Curve for a survival model
        
        Parameters:
        - estimator: The survival model used to make predictions (e.g., CoxPH from sksurv)
        - X: The input data used for making predictions
        - y: A structured array of survival times and event occurrences 
        - y_train: Structured array of survival times and event occurrences for 
            training (observed data)
        """
        self._binary_image = io.BytesIO()
        
        # Compute time-dependent ROC AUC for each time point
        _, event_times = zip(*y)
        
        max_val = max(event_times) - 0.1 if isinstance(max(event_times), float) else max(event_times)
        times = np.arange(min(event_times), max_val)
        

        # Calculate cumulative dynamic AUC (time-dependent ROC AUC)
        aucs, _ = cumulative_dynamic_auc(
            np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')]),
            np.array(y, dtype=[('event', 'bool'), ('time', 'float')]),
            estimator.predict(X),
            times
        )
        
        # Plot the time-dependent ROC AUC over time
        plt.plot(times, aucs)
        plt.xlabel("Temps de suivi")
        plt.ylabel("AUC dynamique cumulative")
        plt.title('ROC dynamique curve')
        plt.axhline(np.nanmean(aucs), color='r', linestyle='--', label=f'Mean AUC = {np.nanmean(aucs):.2f}')
        plt.legend()
        plt.grid(True)

        plt.savefig(self._binary_image, format='png')
        plt.close()

        return self
    
    def suitable(self, type_of_target: str) -> bool:
        """
        Does this plot is usable for a given type_of_target?
        """
        return type_of_target in ['survival']
