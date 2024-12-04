"""
[STEP]  CatBoost Regressor
"""
import textwrap
from typing import Any
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActCatBoostRegressor(Predictor):
    """[STEP]  CatBoost Regressor"""

    name: str = "CatBoost Regressor"
    _description: str = textwrap.dedent('''\
        CatBoostRegressor is a powerful tool that helps computers make accurate
        predictions for continuous outcomes by learning from both positive and
        negative examples simultaneously.''')
    _description_long: str = textwrap.dedent('''\
        CatBoostRegressor is a gradient boosting algorithm specifically
        designed for regression tasks.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2017,
            'name': 'CatBoost: unbiased boosting with categorical features',
            'authors': [
                'Liudmila Prokhorenkova',
                'Gleb Gusev',
                'Aleksandr Vorobev',
                'Anna Veronika Dorogush',
                'Andrey Gulin'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1706.09516',
            'publisher': 'Advances in Neural Information Processing Systems 31 (NeurIPS 2018)'
        }
    ]

    def __init__(self):
        self.configuration = {
            'iterations': {
                'description': 'The maximum number of trees that can be built.',
                'default': 1000,
                'range': [100, 10000]
            },
            'learning_rate': {
                'description': 'The learning rate.',
                'default': 0.03,
                'range': [0.001, 1]
            },
            'depth': {
                'description': 'Depth of the tree.',
                'default': 6,
                'range': [1, 16]
            },
            'l2_leaf_reg': {
                'description': 'Coefficient at the L2 regularization term of the cost function.',
                'default': 3,
                'range': [0, 10]
            },
            'border_count': {
                'description': 'The number of splits for numerical features.',
                'default': 254,
                'range': [1, 255]
            },
            'loss_function': {
                'description': 'The metric to use in training.',
                'default': 'RMSE',
                'categorical': ['RMSE',
                                'MAE',
                                'Quantile',
                                'LogLinQuantile',
                                'Poisson',
                                'MAPE']
                                #, 'Lq']
            },
            'eval_metric': {
                'description': 'The metric to be used for validation data.',
                'default': 'RMSE',
                'categorical': ['RMSE',
                                'MAE',
                                'R2',
                                'Quantile',
                                'LogLinQuantile',
                                'Poisson',
                                'MAPE']
                                # , 'Lq']
            },
            'bootstrap_type': {
                'description': 'The method for sampling the weights of objects.',
                'default': 'Bayesian',
                'categorical': ['Bayesian', 'Bernoulli', 'MVS']
            },
            'leaf_estimation_iterations': {
                'description': 'The number of iterations for leaf estimation.',
                'default': 10,
                'range': [1, 50]
            }
        }
        self.model: CatBoostRegressor = None
        self.label_encoder: LabelEncoder = LabelEncoder()

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = CatBoostRegressor(verbose=0, **self.passthrough_parameters())

        self.label_encoder.fit(dataset.y)
        self.model.fit(dataset.X, self.label_encoder.transform(dataset.y))
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['continuous']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
