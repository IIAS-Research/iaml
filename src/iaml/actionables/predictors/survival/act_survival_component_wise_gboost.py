"""[STEP] Componentwise Gradient Boosting Survival Analysis"""
import textwrap
from typing import Any
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival', 'minimal_predictor')
class ActComponentwiseGradientBoostingSurvivalAnalysis(Predictor):
    """[STEP] Componentwise Gradient Boosting Survival Analysis"""

    name: str = "ComponentwiseGradientBoostingSurvivalAnalysis"
    _usage: str = "Use when you want stagewise boosting with feature selection for survival instead of ActGradientBoostingSurvivalAnalysis. Applicable to tabular censored survival data, especially with many features. Avoid when a linear model like ActCox or ActCoxnetSurvivalAnalysis is preferred."
    _description: str = textwrap.dedent('''\
        ComponentwiseGradientBoostingSurvivalAnalysis is a survival analysis
        algorithm that uses gradient boosting with componentwise (stagewise) updates
        to estimate the survival function over time. This variant of boosting allows
        the model to fit individual components (features) in a stagewise manner,
        making it a more interpretable approach for feature selection and model
        refinement in survival analysis.''')
    _description_long: str = textwrap.dedent('''\
        ComponentwiseGradientBoostingSurvivalAnalysis extends
        gradient boosting for survival analysis by applying updates one component
        (feature) at a time. This approach improves the model's ability to handle
        sparse datasets or datasets with high-dimensional features, where only
        a few variables may have significant effects on survival outcomes.
        It provides a more interpretable framework for survival analysis,
        as each boosting iteration focuses on fitting individual covariates
        rather than combining all features at once. This method is particularly
        suited for feature selection and handling censored survival data.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Survival ensembles',
            'authors': [
                'T. Hothorn',
                'P. B5hlmann',
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
        self.model = ComponentwiseGradientBoostingSurvivalAnalysis(
            **self.passthrough_parameters()
        )
        X, y = dataset.to_survival()
        self.model.fit(X, y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
