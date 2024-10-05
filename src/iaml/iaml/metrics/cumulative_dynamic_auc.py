"""
[METRIC] Cumulative Dynamic AUC for Survival Models
"""
import pandas as pd
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from ..metric import Metric

class CumulativeDynamicAUCMetric(Metric):
    """
    [METRIC] Cumulative Dynamic AUC for Survival Models
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the time-dependent Cumulative Dynamic Area Under the ROC Curve (AUC) \
                for survival models. The time-dependent AUC measures the model’s ability to \
                distinguish between subjects who experience the event before a given time and \
                those who do not, considering censored data. Higher values indicate \
                better discrimination, with 1 being perfect.'

    def __str__(self):
        return 'cumulative_dynamic_auc'
    
    
    @property
    def needed_prediction(self):
        return 'predict_survival_function'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        """
        Is this metric suitable for this candidate? Must be survival analysis.

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Labels (with two columns: 'duration' and 'event')
            type_of_target (str): Type of target (must be 'survival')

        Returns:
            bool: Suitable?
        """
        return type_of_target == 'survival'
        
    def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, 
                y_train: pd.DataFrame = None, **kwargs) -> float:
        """
        Compute the cumulative dynamic AUC using the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted risk scores or survival probabilities
            y_train (pd.DataFrame): Training data (duration and event status)
            time_points (array-like): Time points at which to compute the AUC
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Mean Cumulative Dynamic AUC across the specified time points
        """
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        
        # Extract time from y test
        _, time = zip(*y)
        times = np.arange(min(time), max(time))
        
        # Extract risk score for each time point
        predictions = np.asarray([[fn(t) for t in times] for fn in y_pred])

        # Calculate cumulative dynamic AUC using sksurv function
        all_points, _ = cumulative_dynamic_auc(y_train, y, predictions, times)
        
        # Return mean AUC across time points
        return np.nanmean(all_points)
