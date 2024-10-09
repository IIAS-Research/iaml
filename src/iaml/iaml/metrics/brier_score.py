
"""
[METRIC] Brier Score for Survival Models
"""
import pandas as pd
import numpy as np
from sksurv.metrics import brier_score
from ..metric import Metric
import textwrap

class BrierScoreMetric(Metric):
    """
    [METRIC] Brier Score for Survival Models
    """
    name = textwrap.dedent('Brier Score for Survival Models')
    description = textwrap.dedent('''The Brier Score is a metric used to assess the accuracy of survival models, 
        which predict the likelihood of an event, such as death or disease, occurring within a specific timeframe. 
        It compares the model's probability predictions to actual outcomes, with lower scores indicating better model performance.''')
    description_long = textwrap.dedent('''The Brier Score measures how well survival models predict the probability of an event happening, like survival over time. 
        It calculates the average squared differences between predicted probabilities and actual outcomes 
        (1 for an event occurring, 0 for it not occurring). The score ranges from 0 to 1, where 0 means perfect 
        predictions and 1 means completely inaccurate ones. This metric is valuable because it not only evaluates prediction accuracy 
        but also considers the uncertainty of those predictions. A lower Brier Score indicates a more reliable model, 
        making it a crucial tool for researchers and practitioners in fields like medicine, where accurate survival 
        predictions can significantly impact decision-making.''')
    refs=[
        {
            'year': 1999,
            'name': 'Assessment and comparison of prognostic classification schemes for survival data',
            'authors': [
                'E. Graf',
                'C. Schmoor',
                'W. Sauerbrei',
                'M. Schumacher'
            ],
            'doi': 'https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18%3C2529::AID-SIM274%3E3.0.CO;2-5',
            'publisher': 'Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545'
        }
    ]

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
            
