"""[STEP]  XGBoost Regressor"""
import textwrap
from typing import Any
from xgboost import XGBRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor', 'minimal_predictor')
class ActXGBoostRegressor(Predictor):
    """[STEP]  XGBoost Regressor"""

    name: str = "XGBoost Regressor"
    _description: str = textwrap.dedent('''\
        XGBoost predicts continuous targets using regularized gradient-boosted
        decision trees.''')
    _description_long: str = textwrap.dedent('''\
        Uses XGBoost's histogram tree builder with a squared-error objective.
        Trees are trained sequentially to improve the ensemble's predictions,
        with row and column sampling available to control overfitting.''')
    _usage: str = "Use when you want boosted-tree regression on tabular data, balancing against ActCatBoostRegressor or ActExtraTreesRegressor. Applicable to continuous targets with numeric or encoded categorical features. Avoid when you need native categorical handling or a very fast baseline."
    refs: list[dict[str, Any]] = [
        {
            'name': 'XGBoost: A Scalable Tree Boosting System',
            'year': 2016,
            'authors': [
                'Tianqi Chen',
                'Carlos Guestrin'
            ],
            'doi': 'https://doi.org/10.1145/2939672.2939785',
            'publisher': 'ACM SIGKDD 2016, pages 785--794'
        }
    ]
    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 6,
                'range': [1, 16]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 0.1,
                'range': [0.001, 1.0]
            },
            'subsample': {
                'description': 'Fraction of training rows sampled for each tree',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, 500]
            },
            'min_child_weight': {
                'description': 'Minimum sum of instance Hessians required in a child',
                'default': 1.0,
                'range': [0.0, 20.0]
            },
            'colsample_bytree': {
                'description': 'Fraction of feature columns sampled for each tree',
                'default': 1.0,
                'range': [0.1, 1.0]
            }
        }
        self.model: XGBRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = XGBRegressor(
            **self.passthrough_parameters(),
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=1,
        )
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['continuous']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
