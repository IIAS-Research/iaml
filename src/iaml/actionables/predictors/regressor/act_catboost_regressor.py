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
                'range': [300, 10000],
            },
            'learning_rate': {
                'description': 'The learning rate.',
                'default': 0.03,
                'range': [0.005, 0.20],
            },
            'depth': {
                'description': 'Depth of the tree.',
                'default': 6,
                'range': [3, 10],
            },
            'grow_policy': {
                'description': 'The tree growing policy.',
                'default': 'SymmetricTree',
                'categorical': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            },
            'l2_leaf_reg': {
                'description': 'Coefficient of L2 regularization on leaf values.',
                'default': 3.0,
                'range': [1e-3, 1e3],
            },
            'random_strength': {
                'description': 'Randomness added when scoring potential splits.',
                'default': 1.0,
                'range': [0.0, 5.0],
            },
            'min_data_in_leaf': {
                'description': 'Minimum sample count in a leaf eligible for splitting.',
                'default': 1,
                'range': [1, 300],
            },
            'loss_function': {
                'description': 'The regression objective, fixed during automatic search.',
                'default': 'RMSE',
            },
            'eval_metric': {
                'description': 'Validation metric; None keeps the CatBoost default.',
                'default': None,
            },
            'border_count': {
                'description': 'Numeric split count; None keeps the CatBoost default.',
                'default': None,
            },
            'bootstrap_type': {
                'description': 'Weight sampling method; None keeps the CatBoost default.',
                'default': None,
            },
            'leaf_estimation_iterations': {
                'description': 'Leaf estimation iterations; None keeps the CatBoost default.',
                'default': None,
            },
        }

        self.model: CatBoostRegressor = None

    def passthrough_parameters(self, default: bool = True) -> dict[str, Any]:
        """Keep CatBoost defaults and omit leaf size for symmetric trees."""
        parameters = {key: value for key, value in super().passthrough_parameters(default).items()
                      if value is not None}
        if self.get_config('grow_policy') == 'SymmetricTree':
            parameters.pop('min_data_in_leaf', None)
        return parameters

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
