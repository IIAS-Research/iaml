"""
[STEP]  CatBoost Regressor
"""
import textwrap
from typing import Any
from catboost import CatBoostRegressor, CatBoostError
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step
from ....logger import Logger

@is_step('predictor', 'tabular', 'regressor', 'minimal_predictor')
class ActCatBoostRegressor(Predictor):
    """[STEP]  CatBoost Regressor"""

    name: str = "CatBoost Regressor"
    _usage: str = "Use when you need strong tabular regression with categorical features versus ActDecisionTreeRegressor. Applicable to continuous targets with mixed numeric/categorical columns. Avoid when interpretability, ultra-low latency, or tiny data dominate."
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
                'range': [0.001, 1.0]
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

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = CatBoostRegressor(verbose=0, **self.passthrough_parameters())
        try:
            self.model.fit(dataset.X, dataset.y)
        except CatBoostError as exc:
            self._log_failure(dataset, exc)
            raise ValueError(f"CatBoostRegressor training failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            self._log_failure(dataset, exc)
            raise
        return self

    def _log_failure(self, dataset: Dataset, exc: Exception) -> None:
        """Log enriched debug info when CatBoost crashes."""
        shape = getattr(dataset.X, "shape", None)
        message = (
            "[CatBoostRegressor] crash detected "
            f"(shape={shape}, target_len={len(dataset.y)}, "
            f"params={self.passthrough_parameters()}): {exc}"
        )
        logger = Logger()
        if logger.verbose <= 3 and logger.verbose != -1:
            logger.console.log(message)
        logger.error(message)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['continuous']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
