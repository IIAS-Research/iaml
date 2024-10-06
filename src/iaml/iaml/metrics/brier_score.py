
"""
[METRIC] Brier Score for Survival Models
"""
import pandas as pd
import numpy as np
from sksurv.metrics import brier_score
from ..metric import Metric


class BrierScoreMetric(Metric):
    """
    [METRIC] Brier Score for Survival Models
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the Brier Score, which measures the accuracy of probabilistic \
                predictions for survival models. The Brier Score is the mean squared error \
                between the predicted probabilities and the actual outcomes, with adjustments \
                for censored data. Lower values indicate better accuracy, with 0 being perfect.'

    def __str__(self):
        return 'brier_score'
    
    
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
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, 
                y_train:pd.DataFrame=None, **kwargs) -> float:
        """
        Compute the Brier Score  with the predicted data.

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
        
        
        time = max(time)
        if isinstance(time, float):
            time -= 0.1
            
        predictions = [fn(time) for fn in y_pred]
        
        # Calculate integrated Brier score using sksurv function
        return brier_score(y_train, y, predictions, time)[1][0]
            
