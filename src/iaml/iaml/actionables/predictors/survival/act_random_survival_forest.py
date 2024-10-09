"""
[STEP] Learn : Random Survival Forest
"""
from sksurv.ensemble import RandomSurvivalForest
import numpy as np

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step
import textwrap

@is_step('predictor', 'tabular', 'survival')
class ActRandomSurvivalForest(Predictor):
    """
    [STEP] Learn : Random Survival Forest
    """
    name = textwrap.dedent("Learn : RandomSurvivalForest")
    description = textwrap.dedent('''RandomSurvivalForest is a survival analysis algorithm 
        that uses an ensemble of decision trees to estimate the survival function 
        over time. It is a non-parametric model that handles complex relationships 
        and can model non-linear effects of covariates.''')
    description_long = textwrap.dedent('''RandomSurvivalForest is a flexible survival analysis 
        algorithm that uses an ensemble of decision trees to predict the time 
        until an event occurs. It extends the concept of random forests to survival 
        data, handling complex interactions and non-linear relationships between 
        input features (covariates). Unlike parametric models such as the Cox 
        Proportional Hazards model, RandomSurvivalForest makes fewer assumptions 
        about the underlying data, making it useful in cases where the assumptions 
        of proportional hazards do not hold. It also efficiently manages censored 
        data, where the event may not have occurred during the study period.''')
    
    refs = [
        {
            'year': 2008,
            'name': 'Random survival forests',
            'authors': [
                'H. Ishwaran',
                'U. B. Kogalur',
                'E. H. Blackstone',
                'M. S. Lauer'
            ],
            'doi': 'https://doi.org/10.1214/08-AOAS169',
            'publisher': 'The Annals of Applied Statistics, 2(3): page 841-860'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'n_estimators': {
                'description': 'Number of trees in the forest.',
                'default': 100,
                'range': [1, 1000],
                'passthrough': False
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node.',
                'default': 2,
                'range': [2, 20],
                'passthrough': False
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 20],
                'passthrough': False
            },
            'max_depth': {
                'description': 'The maximum depth of the tree. If None, then nodes are expanded \
                    until all leaves are pure.',
                'default': None,
                'range': [1, None],
                'passthrough': False
            }
        }
        self.model: RandomSurvivalForest = None
        
    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        """
        Fit Random Survival Forest on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = RandomSurvivalForest(
            **self.passthrough_parameters()
        )
        
        y = np.array(dataset.y, dtype=[('event', 'bool'), ('time', 'float')])
        self.model.fit(dataset.X, y)
        
        return self
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5  # neutral
