"""[STEP] LightGBM Classifier"""
import textwrap
from typing import Any

try:
    from lightgbm import LGBMClassifier
    from lightgbm.basic import LightGBMError
    _LGBM_ERRORS: tuple[type[Exception], ...] = (LightGBMError,)
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore
    _LGBM_ERRORS = tuple()

from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActLightGBMClassifier(Predictor):
    """[STEP] LightGBM Classifier"""

    name: str = "LightGBM Classifier"
    _description: str = textwrap.dedent('''\
        LightGBMClassifier is a gradient boosting algorithm that builds
        decision trees efficiently for classification tasks.''')
    _description_long: str = textwrap.dedent('''\
        LightGBMClassifier trains an ensemble of decision trees using histogram-based
        splits and leaf-wise growth. It is designed to be fast while preserving
        accuracy on tabular classification problems.''')
    _usage: str = "Use when you want fast gradient boosting on tabular classification; compare to ActCatBoost or ActExtraTreesClassifier. Applicable to numeric and categorical features with binary or multiclass targets. Avoid when you need a simple, interpretable model or very small data."
    refs: list[dict[str, Any]] = [
        {
            'year': 2017,
            'name': 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
            'authors': [
                'Guolin Ke',
                'Qi Meng',
                'Thomas Finley',
                'Taifeng Wang',
                'Wei Chen',
                'Weidong Ma',
                'Qiwei Ye',
                'Tie-Yan Liu'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.1712.01005',
            'publisher': 'Advances in Neural Information Processing Systems 30 (NeurIPS 2017)'
        }
    ]

    def __init__(self):
        self.configuration = {
            'boosting_type': {
                'description': 'Type of boosting algorithm.',
                'default': 'gbdt',
                'categorical': ['gbdt', 'dart']
            },
            'n_estimators': {
                'description': 'Number of boosting iterations.',
                'default': 200,
                'range': [50, 1000]
            },
            'learning_rate': {
                'description': 'Shrinkage rate applied to each tree.',
                'default': 0.1,
                'range': [0.01, 1.0]
            },
            'num_leaves': {
                'description': 'Maximum number of leaves in one tree.',
                'default': 31,
                'range': [7, 255]
            },
            'max_depth': {
                'description': 'Maximum depth of a tree, -1 means no limit.',
                'default': -1,
                'categorical': [-1, 3, 5, 10, 15]
            },
            'min_child_samples': {
                'description': 'Minimum number of data in one leaf.',
                'default': 20,
                'range': [5, 200]
            },
            'subsample': {
                'description': 'Fraction of data to use for each boosting iteration.',
                'default': 1.0,
                'range': [0.5, 1.0]
            },
            'subsample_freq': {
                'description': 'Frequency for subsampling, 0 means disabled.',
                'default': 0,
                'range': [0, 10]
            },
            'colsample_bytree': {
                'description': 'Fraction of features used for each tree.',
                'default': 1.0,
                'range': [0.5, 1.0]
            },
            'reg_alpha': {
                'description': 'L1 regularization.',
                'default': 0.0,
                'range': [0.0, 1.0]
            },
            'reg_lambda': {
                'description': 'L2 regularization.',
                'default': 0.0,
                'range': [0.0, 1.0]
            },
            'class_weight': {
                'description': textwrap.dedent('''\
                    The "balanced" mode uses values of y to adjust weights inversely
                    proportional to class frequencies.'''),
                'default': None,
                'categorical': [None, 'balanced']
            },
            'random_state': {
                'description': 'Random seed for reproducibility.',
                'default': 42
            },
            'verbosity': {
                'description': 'Controls the level of LightGBM verbosity.',
                'default': -1,
                'categorical': [-1, 0, 1]
            }
        }
        self.model: LGBMClassifier = None
        self.columns: list[str] = []
        self.categorical_columns: list[str] = []
        self._category_levels: dict[str, list] = {}

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _prepare_features(self, X, fit: bool = False):
        X_selected = self._select_features(X)
        if not hasattr(X_selected, 'copy'):
            return X_selected
        X_prepared = X_selected.copy()
        if self.categorical_columns:
            for column in self.categorical_columns:
                if column not in X_prepared.columns:
                    continue
                X_prepared[column] = X_prepared[column].astype('category')
                if not fit and column in self._category_levels:
                    X_prepared[column] = X_prepared[column].cat.set_categories(
                        self._category_levels[column]
                    )
        if fit:
            self._category_levels = {
                column: list(X_prepared[column].cat.categories)
                for column in self.categorical_columns
                if column in X_prepared.columns and hasattr(X_prepared[column], 'cat')
            }
        return X_prepared

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is required for ActLightGBMClassifier. "
                "Install with: pip install lightgbm"
            )

        self.columns = dataset.get_columns_names_by_type(
            [DataType.NUMERIC, DataType.CATEGORICAL]
        )
        if not self.columns:
            self.columns = dataset.features
        self.categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)

        X_prepared = self._prepare_features(dataset.X, fit=True)
        self.model = LGBMClassifier(**self.passthrough_parameters())

        categorical_features = []
        if self.categorical_columns and hasattr(X_prepared, 'columns'):
            categorical_features = [
                col for col in self.categorical_columns if col in X_prepared.columns
            ]

        try:
            if categorical_features:
                self.model.fit(
                    X_prepared,
                    dataset.y,
                    categorical_feature=categorical_features
                )
            else:
                self.model.fit(X_prepared, dataset.y)
        except _LGBM_ERRORS as exc:
            raise ValueError(f"LightGBMClassifier training failed: {exc}") from exc
        return self

    def predict(self, X):
        return super().predict(self._prepare_features(X))

    def predict_proba(self, X):
        return super().predict_proba(self._prepare_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._prepare_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        supported = dataset.get_columns_names_by_type(
            [DataType.NUMERIC, DataType.CATEGORICAL]
        )
        return LGBMClassifier is not None and dataset.type_of_target in \
            ['binary', 'multiclass', 'multilabel-indicator'] and bool(supported)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
