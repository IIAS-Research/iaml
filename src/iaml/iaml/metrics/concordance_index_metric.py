import pandas as pd
from sksurv.metrics import concordance_index_censored
from ..metric import Metric

class ConcordanceIndexMetric(Metric):
    """
    [METRIC] Concordance Index for Survival Models using sksurv
    """
    def explain(self) -> str:
        """Describe metric

        Returns:
            str: Metric description
        """
        return 'Computes the concordance index using sksurv, a common metric used to evaluate \
                the prediction accuracy of survival models. It measures the proportion of \
                correctly ordered event times predicted by the model.'

    def __str__(self):
        return 'concordance_index'

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
        
    def compute(self, y:pd.DataFrame, y_pred:pd.DataFrame, **kwargs) -> float:
        """
        Compute the concordance index with the predicted data.

        Args:
            y (pd.DataFrame): Ground truth data (duration and event status)
            y_pred (pd.DataFrame): Predicted data (risk scores or predicted survival times)

        Returns:
            float: Computed concordance index
        """
        event, time = zip(*y)
        
        # Calculate concordance index using sksurv function
        result = concordance_index_censored(event, time, y_pred)
        
        return result[0]  # The first value in the result is the concordance index
