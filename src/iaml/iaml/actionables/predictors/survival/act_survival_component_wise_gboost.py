"""
[STEP] Learn : Componentwise Gradient Boosting Survival Analysis
"""
import textwrap
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
import numpy as np

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'survival')
class ActComponentwiseGradientBoostingSurvivalAnalysis(Predictor):
    """
    [STEP] Learn : Componentwise Gradient Boosting Survival Analysis
    """
    name = "Learn : ComponentwiseGradientBoostingSurvivalAnalysis"
    description = textwrap.dedent('''\
        ComponentwiseGradientBoostingSurvivalAnalysis is a survival analysis 
        algorithm that uses gradient boosting with componentwise (stagewise) updates 
        to estimate the survival function over time. This variant of boosting allows 
        the model to fit individual components (features) in a stagewise manner, 
        making it a more interpretable approach for feature selection and model 
        refinement in survival analysis.''')
    description_long = textwrap.dedent('''\
        ComponentwiseGradientBoostingSurvivalAnalysis extends 
        gradient boosting for survival analysis by applying updates one component 
        (feature) at a time. This approach improves the model's ability to handle 
        sparse datasets or datasets with high-dimensional features, where only 
        a few variables may have significant effects on survival outcomes. 
        It provides a more interpretable framework for survival analysis, 
        as each boosting iteration focuses on fitting individual covariates 
        rather than combining all features at once. This method is particularly 
        suited for feature selection and handling censored survival data.''')
    
    refs = [
        {
            'year': 2006,
            'name': 'Survival ensembles',
            'authors': [
                'T. Hothorn',
                'P. Bühlmann',
                'S. Dudoit',
                'A. Molinaro',
                'M. J. van der Laan'
            ],
            'doi': 'https://doi.org/10.1093/biostatistics/kxj011',
            'publisher': 'Biostatistics, 7(3), 355-373'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'n_estimators': {
                'description': 'Number of boosting stages to be run.',
                'default': 100,
                'range': [1, 1000],
                'passthrough': False
            },
            'learning_rate': {
                'description': 'Learning rate shrinks the contribution of each tree by this value.',
                'default': 0.1,
                'range': [0.01, 1.0],
                'passthrough': False
            },
            'max_depth': {
                'description': 'The maximum depth of the individual trees.',
                'default': 1,
                'range': [1, 10],
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
            }
        }
        self.model: ComponentwiseGradientBoostingSurvivalAnalysis = None
        
    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        """
        Fit Componentwise Gradient Boosting Survival Analysis on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = ComponentwiseGradientBoostingSurvivalAnalysis(
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
