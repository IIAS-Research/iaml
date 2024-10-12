"""
[METRIC] Concordance Index for Survival Models using sksurv
"""
import textwrap
import pandas as pd
from sksurv.metrics import concordance_index_censored
from ..metric import Metric

class ConcordanceIndexMetric(Metric):
    """
    [METRIC] Concordance Index for Survival Models using sksurv
    """
    name = 'Concordance Index for Survival Models using sksurv'
    description = textwrap.dedent('''\
        The Concordance Index for Survival Models using sksurv measures how well 
        a survival model predicts the order of events, such as survival times. A higher index 
        value indicates better predictive accuracy.''')
    description_long = textwrap.dedent('''\
        The Concordance Index for Survival Models using sksurv evaluates the performance of 
        survival models by assessing their ability to correctly rank individuals based on their 
        survival times. It focuses on the relative timing of events rather than exact predictions. 
        The index ranges from 0 to 1, where 0.5 indicates no predictive ability (similar to random guessing) 
        and 1 indicates perfect prediction of event order. This metric is particularly useful in survival 
        analysis, as it helps researchers and clinicians understand how well their models perform in 
        predicting outcomes, making it a valuable tool in fields like healthcare and clinical research.
        ''')
    
    refs=[
        {
            'year': 1996,
            'name': textwrap.dedent("""\
                Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors
                """),
            'authors': [
                'FRANK E.',
                'HARRELL Jr.',
                'KERRY L',
                'LEE',
                'DANIEL B. MARK'
            ],
            'doi': textwrap.dedent("""\
                https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4
                """),
            'publisher': 'Statistics in Medicine, 15(4), 361-87'
        }
    ]

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
