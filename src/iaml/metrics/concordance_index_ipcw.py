"""[METRIC] Concordance Index with Inverse Probability of Censoring 
Weights (IPCW) for Survival Models
"""
from typing import Any
import textwrap
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from ..metric import Metric
from ..dataset import Dataset

class ConcordanceIndexIPCWMetric(Metric):
    """[METRIC] Concordance Index with Inverse Probability of Censoring 
    Weights (IPCW) for Survival Models
    """
    name: str = 'C-Index IPC'
    _description: str = textwrap.dedent('''\
        The Concordance Index with Inverse Probability of Censoring Weights (IPCW) 
        evaluates survival models by measuring how well they predict the order of events, 
        like survival times, while accounting for censored data. A higher index indicates 
        better predictive accuracy.''')
    _description_long: str = textwrap.dedent('''\
        The Concordance Index with Inverse Probability of Censoring Weights (IPCW) 
        assesses survival models by focusing on the ranking of survival times. It addresses the issue of censored 
        data—when some outcomes are not fully observed—by applying weights based on the probability of censoring. 
        The index ranges from 0 to 1, with 0.5 indicating no predictive ability and 1 indicating perfect prediction. 
        By incorporating IPCW, this metric provides a more accurate evaluation of a model's performance, making it 
        essential for researchers in fields like medicine and epidemiology.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2011,
            'name': textwrap.dedent("""\
                On the C-statistics for evaluating overall adequacy of risk prediction 
                procedures with censored survival data"""),
            'authors': [
                'Hajime Uno',
                'Tianxi Cai',
                'Michael J. Pencina',
                'Ralph B. D\'Agostino',
                'L. J. Wei'
            ],
            'doi': 'https://doi.org/10.1002/sim.4154',
            'publisher': 'Statistics in Medicine, vol. 18, no. 17-18, pp. 2529-2545'
        }
    ]

    def __str__(self) -> str:
        return 'concordance_index_ipcw'

    @property
    def needed_prediction(self) -> str:
        return 'predict'

    def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
        return type_of_target == 'survival'

    def compute(
        self,
        y: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_train: pd.DataFrame = None,
        **kwargs) -> float:
        """Compute the Concordance Index (C-index) with IPCW using the predicted data.
        
        :param pd.DataFrame y: Ground truth to compute the metric.
        :param pd.DataFrame y_pred: Prediction to compute the metric.
        :param pd.DataFrame y_train: Training ground truth. Default to None.
        :param dict, optional \\**kwargs: Additional parameters
        :return: Computed value
        """
        y_train = np.array(y_train, dtype=[('event', 'bool'), ('time', 'float')])
        y = Dataset.fix_y_survival(y, y_train)
        y = np.array(y, dtype=[('event', 'bool'), ('time', 'float')])

        # Calculate concordance index using sksurv function
        return concordance_index_ipcw(y_train, y, y_pred)[0]
