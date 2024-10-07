"""
[METRIC] Integrated Brier Score for Survival Models
"""
import pandas as pd
import numpy as np
from sksurv.metrics import integrated_brier_score
from ..metric import Metric

class IntegratedBrierScoreMetric(Metric):
    """
    [METRIC] Integrated Brier Score for Survival Models
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the Integrated Brier Score (IBS) using sksurv, which evaluates the \
                prediction accuracy of a survival model by comparing the predicted probabilities \
                of survival with the actual survival status over time. Lower values indicate better \
                model performance. 0.25 is considered neutral for a balanced binary classification \
                problem.'

    def __str__(self):
        return 'integrated_brier_score'
    
    
    @property
    def needed_prediction(self):
        return 'predict_survival_function'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        """
        Is this metric suitable for this candidate? Must be survival analysis.

        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): labels (with two columns: 'duration' and 'event')
            type_of_target (str): Type of target (must be 'survival')

        Returns:
            bool: Suitable?
        """
        return type_of_target == 'survival'
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, 
                y_train:pd.DataFrame=None, **kwargs) -> float:
        """
        Compute the Integrated Brier Score (IBS) with the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted data (risk scores or predicted survival times)
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Computed Integrated Brier Score
        """
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        
        # Extract time from y test
        _, time = zip(*y)
        times = np.arange(min(time), max(time))
        
        # Extract risk score for each time point
        predictions = np.asarray([[fn(t) for t in times] for fn in y_pred])

        # Calculate integrated Brier score using sksurv function
        return integrated_brier_score(y_train, y, predictions, times)
        
