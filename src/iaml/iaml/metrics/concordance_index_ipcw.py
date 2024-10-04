import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from ..metric import Metric

class ConcordanceIndexIPCWMetric(Metric):
    """
    [METRIC] Concordance Index with Inverse Probability of Censoring Weights (IPCW) for Survival Models
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the Concordance Index (C-index) using IPCW, which measures the predictive accuracy \
                of a survival model. The C-index is the proportion of all pairs of subjects whose predicted \
                survival times are correctly ordered. IPCW adjusts for censored data. A value of 0.5 indicates \
                random predictions, and a value of 1 indicates perfect predictions.'

    def __str__(self):
        return 'concordance_index_ipcw'
    
    
    @property
    def needed_prediction(self):
        return 'predict'

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
        Compute the Concordance Index (C-index) with IPCW using the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted data (risk scores or predicted survival times)
            kwargs: Additional arguments, including y_train and X_train

        Returns:
            float: Computed Concordance Index (C-index) using IPCW
        """
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        
        # Calculate concordance index using sksurv function
        return concordance_index_ipcw(y_train, y, y_pred)[0]
