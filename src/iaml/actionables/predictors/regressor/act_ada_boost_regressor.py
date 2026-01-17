"""[STEP]  AdaBoost Regressor"""
import textwrap
from typing import Any
from sklearn.ensemble import AdaBoostRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActAdaBoostRegressor(Predictor):
    """[STEP]  AdaBoost Regressor"""

    name: str = "AdaBoost Regressor"
    _description: str = textwrap.dedent('''\
        AdaBoostRegressor is a powerful tool that combines many simple models
        to make accurate predictions for continuous outcomes.''')
    _description_long: str = textwrap.dedent('''\
        AdaBoostRegressor is an ensemble learning technique
        used for regression problems. It works by combining multiple weak learners
        (simple models) into a strong learner.''')
    _usage: str = "Use when regression needs boosting and you want an alternative to ActDecisionTreeRegressor or ActExtraTreesRegressor. Applicable to continuous targets with modest features and nonlinear signal. Avoid when data is very noisy, high-dimensional, or you need strong interpretability."
    refs: list[dict[str, Any]] = [
        {
            'year': 1995,
            'name': (
                'A desicion-theoretic generalization of on-line learning '
                'and an application to boosting'
            ),
            'authors': [
                'Yoav Freund',
                'Robert E. Schapire'
            ],
            'doi': 'https://doi.org/10.1007/3-540-59119-2_166',
            'publisher': 'Springer, Berlin, Heidelberg'
        }
    ]
    def __init__(self):
        self.configuration = {
            'learning_rate': {
                'description': 'Weight applied to each regressor at each boosting iteration',
                'default': 0.1,
                'range': [0.01, 2.0]
            },
            'loss': {
                'description': textwrap.dedent('''\
                    The loss function to use when updating the weights after
                    each boosting iteration.'''),
                'default': "linear",
                'categorical': ["linear", "square", "exponential"]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 50,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model: AdaBoostRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = AdaBoostRegressor(**self.passthrough_parameters())

        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
