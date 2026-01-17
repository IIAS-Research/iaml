"""[STEP] Select From Model."""
import textwrap

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_selection')
class ActSelectFromModel(Actionable):
    """[STEP] Select From Model."""

    name: str = 'Select From Model'
    _description: str = textwrap.dedent('''\
        Select numeric features using a {estimator} model and an
        importance threshold of {threshold}.''')
    _description_long: str = textwrap.dedent('''\
        SelectFromModel trains a base estimator and removes features with low
        importance (tree-based) or low absolute coefficients (L1-regularized
        linear models). This step supports L1 and tree estimators to reduce
        dimensionality and keep the most informative predictors.''')
    _usage: str = 'Use when you need model-based selection via L1 or trees over ActSelectKBest. Applicable to supervised numeric features for regression or classification. Avoid when no target, mostly categorical data, or you prefer ActRemoveLowVarianceColumn.'

    def __init__(self):
        self.configuration = {
            'estimator': {
                'description': 'Base estimator type used for feature importance.',
                'default': 'l1',
                'categorical': ['l1', 'tree']
            },
            'threshold': {
                'description': textwrap.dedent('''\
                    Importance threshold. Accepts "mean", "median", or a numeric
                    value.'''),
                'default': 'median'
            },
            'max_features': {
                'description': 'Maximum number of features to keep (None for no limit).',
                'default': None
            },
            'l1_alpha': {
                'description': 'Regularization strength for Lasso (regression).',
                'default': 0.01,
                'range': [1e-4, 10.0]
            },
            'l1_C': {
                'description': 'Inverse regularization strength for LogisticRegression.',
                'default': 1.0,
                'range': [0.01, 100.0]
            },
            'tree_n_estimators': {
                'description': 'Number of trees in the ensemble.',
                'default': 50,
                'range': [10, 500]
            },
            'tree_max_depth': {
                'description': 'Maximum depth of each tree.',
                'default': 10,
                'range': [1, 100]
            },
            'tree_min_samples_leaf': {
                'description': 'Minimum number of samples required at a leaf node.',
                'default': 1,
                'range': [1, 20]
            },
            'random_state': {
                'description': 'Random seed for estimators that support it.',
                'default': 42
            }
        }

        self.optimizable: bool = True
        self.columns: list[str] = []
        self.selected_columns: list[str] = []
        self.columns_to_drop: list[str] = []
        self.importances: dict[str, float] = {}
        self.selector: SelectFromModel | None = None
        self.threshold_value: float | None = None

    def _resolve_threshold(self):
        threshold = self.get_config('threshold')
        if threshold is None:
            return None

        if isinstance(threshold, str):
            value = threshold.strip().lower()
            if value in ['mean', 'median']:
                return value
            try:
                return float(value)
            except ValueError:
                return None

        try:
            return float(threshold)
        except (TypeError, ValueError):
            return None

    def _resolve_max_features(self, n_features: int) -> int | None:
        value = self.get_config('max_features')
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ['none', '']:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value < 1:
            return None
        return min(value, n_features)

    def _resolve_positive_float(self, key: str) -> float | None:
        value = self.get_config(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def _resolve_positive_int(self, key: str) -> int | None:
        value = self.get_config(key)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value < 1:
            return None
        return value

    def _resolve_max_depth(self) -> int | None:
        value = self.get_config('tree_max_depth')
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ['none', '']:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value < 1:
            return None
        return value

    def _build_estimator(self, dataset: Dataset):
        estimator_type = self.get_config('estimator')
        random_state = self.get_config('random_state')

        if dataset.type_of_target == 'continuous':
            if estimator_type == 'l1':
                alpha = self._resolve_positive_float('l1_alpha')
                if alpha is None:
                    return None
                return Lasso(alpha=alpha, max_iter=2000)
            if estimator_type == 'tree':
                n_estimators = self._resolve_positive_int('tree_n_estimators')
                min_samples_leaf = self._resolve_positive_int('tree_min_samples_leaf')
                max_depth = self._resolve_max_depth()
                if None in [n_estimators, min_samples_leaf]:
                    return None
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
            return None

        if dataset.type_of_target in ['binary', 'multiclass']:
            if estimator_type == 'l1':
                c_value = self._resolve_positive_float('l1_C')
                if c_value is None:
                    return None
                return LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    max_iter=1000,
                    random_state=random_state,
                    C=c_value,
                    multi_class='ovr'
                )
            if estimator_type == 'tree':
                n_estimators = self._resolve_positive_int('tree_n_estimators')
                min_samples_leaf = self._resolve_positive_int('tree_min_samples_leaf')
                max_depth = self._resolve_max_depth()
                if None in [n_estimators, min_samples_leaf]:
                    return None
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
        return None

    def _extract_importances(self, estimator, columns: list[str]) -> dict[str, float]:
        if estimator is None or not columns:
            return {}

        values = None
        if hasattr(estimator, 'coef_'):
            coefs = np.asarray(estimator.coef_)
            if coefs.ndim == 1:
                values = np.abs(coefs)
            else:
                values = np.mean(np.abs(coefs), axis=0)
        elif hasattr(estimator, 'feature_importances_'):
            values = np.asarray(estimator.feature_importances_)

        if values is None:
            return {}

        return {
            column: float(value)
            for column, value in zip(columns, values)
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.selected_columns = []
        self.columns_to_drop = []
        self.importances = {}
        self.selector = None
        self.threshold_value = None
        self.explanations = []

        if dataset.y is None or dataset.type_of_target is None:
            return self

        if not self.columns:
            return self

        threshold = self._resolve_threshold()
        if threshold is None:
            return self

        estimator = self._build_estimator(dataset)
        if estimator is None:
            return self

        max_features = self._resolve_max_features(len(self.columns))
        if max_features is not None and max_features != self.get_config('max_features'):
            self.configure('max_features', max_features)  # pylint: disable=too-many-function-args

        self.selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold,
            max_features=max_features
        )
        self.selector.fit(dataset.X[self.columns], dataset.y)

        support = self.selector.get_support()
        self.selected_columns = list(pd.Index(self.columns)[support])
        self.columns_to_drop = list(pd.Index(self.columns)[~support])

        self.importances = self._extract_importances(
            getattr(self.selector, 'estimator_', None),
            self.columns
        )
        self.threshold_value = getattr(self.selector, 'threshold_', None)

        if self.columns_to_drop:
            threshold_display = threshold
            if isinstance(self.threshold_value, (float, int)):
                threshold_display = f"{self.threshold_value:.6g}"
            for column in self.columns_to_drop:
                importance = self.importances.get(column)
                if importance is None or np.isnan(importance):
                    self.explanations.append(
                        f"Dropped column **`{column}`** because its importance "
                        f"was below the threshold ({threshold_display})."
                    )
                else:
                    self.explanations.append(
                        f"Dropped column **`{column}`** because its importance "
                        f"({importance:.6g}) was below the threshold ({threshold_display})."
                    )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that were not selected.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        if not self.columns_to_drop:
            return X

        drop_cols = [column for column in self.columns_to_drop if column in X.columns]
        if not drop_cols:
            return X
        return X.drop(columns=drop_cols)

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.y is None or dataset.type_of_target is None:
            return False

        if dataset.type_of_target == 'survival':
            return False

        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False

        if self._resolve_threshold() is None:
            return False

        if self._build_estimator(dataset) is None:
            return False

        return True

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
