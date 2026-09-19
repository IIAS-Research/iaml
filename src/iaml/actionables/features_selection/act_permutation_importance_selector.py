"""[STEP] Permutation Importance Selector."""
import textwrap

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_selection')
class ActPermutationImportanceSelector(Actionable):
    """[STEP] Permutation Importance Selector."""

    name: str = 'Permutation Importance Selector'
    _description: str = textwrap.dedent('''\
        Select numeric features with permutation importance above {threshold}
        using a {estimator} estimator.''')
    _description_long: str = textwrap.dedent('''\
        Permutation importance measures the decrease in model performance when
        a single feature's values are randomly shuffled. Features that cause
        little or no drop in score are considered less informative and can be
        removed to simplify the model.''')
    _usage: str = 'Use when you need post-fit, model-agnostic importance on numeric features; compare ActSelectFromModel for embedded selection. Applicable to supervised numeric targets with a supported estimator. Avoid when data is tiny or you need filters like ActRemoveLowVarianceColumn.'

    def __init__(self):
        self.configuration = {
            'estimator': {
                'description': 'Base estimator used to compute permutation importance.',
                'default': 'tree',
                'categorical': ['linear', 'tree']
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
            'n_repeats': {
                'description': 'Number of permutations used to estimate importance.',
                'default': 5,
                'range': [1, 50]
            },
            'scoring': {
                'description': 'Scoring metric ("auto" to use estimator.score).',
                'default': 'auto'
            },
            'random_state': {
                'description': 'Random seed for estimator and permutations.',
                'default': 42
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
            }
        }

        self.optimizable: bool = True
        self.columns: list[str] = []
        self.selected_columns: list[str] = []
        self.columns_to_drop: list[str] = []
        self.importances: dict[str, float] = {}
        self.importances_std: dict[str, float] = {}
        self.threshold_value: float | None = None
        self.estimator = None

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

    def _resolve_n_repeats(self) -> int | None:
        value = self.get_config('n_repeats')
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value < 1:
            return None
        return value

    def _resolve_scoring(self) -> str | None:
        scoring = self.get_config('scoring')
        if scoring is None:
            return None
        if isinstance(scoring, str):
            value = scoring.strip()
            if value == '' or value.lower() in ['auto', 'none']:
                return None
            return value
        return None

    def _threshold_config_valid(self) -> bool:
        threshold = self.get_config('threshold')
        if threshold is None:
            return True
        if isinstance(threshold, str):
            value = threshold.strip().lower()
            if value in ['mean', 'median', 'none', '']:
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False
        try:
            float(threshold)
            return True
        except (TypeError, ValueError):
            return False

    def _resolve_threshold(self, importances: np.ndarray) -> float | None:
        threshold = self.get_config('threshold')
        if threshold is None:
            return None
        if isinstance(threshold, str):
            value = threshold.strip().lower()
            if value in ['none', '']:
                return None
            if value == 'mean':
                if importances.size == 0:
                    return None
                return float(np.nanmean(importances))
            if value == 'median':
                if importances.size == 0:
                    return None
                return float(np.nanmedian(importances))
            try:
                return float(value)
            except ValueError:
                return None
        try:
            return float(threshold)
        except (TypeError, ValueError):
            return None

    def _build_estimator(self, dataset: Dataset):
        estimator_type = self.get_config('estimator')
        random_state = self.get_config('random_state')

        if dataset.type_of_target in ['continuous', 'continuous-multioutput']:
            if estimator_type == 'linear':
                return LinearRegression()
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
                    random_state=random_state,
                    n_jobs=1
                )
            return None

        if dataset.type_of_target in ['binary', 'multiclass']:
            if estimator_type == 'linear':
                return LogisticRegression(
                    solver='liblinear',
                    max_iter=1000,
                    random_state=random_state,
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
                    random_state=random_state,
                    n_jobs=1
                )
            return None

        return None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.selected_columns = []
        self.columns_to_drop = []
        self.importances = {}
        self.importances_std = {}
        self.threshold_value = None
        self.estimator = None
        self.explanations = []

        if dataset.y is None or dataset.type_of_target is None:
            return self

        if dataset.type_of_target == 'survival':
            return self

        if not self.columns or dataset.X.empty:
            return self

        estimator = self._build_estimator(dataset)
        if estimator is None:
            return self

        n_repeats = self._resolve_n_repeats()
        if n_repeats is None:
            return self

        scoring = self._resolve_scoring()

        max_features = self._resolve_max_features(len(self.columns))
        if max_features is not None and max_features != self.get_config('max_features'):
            self.configure('max_features', max_features)  # pylint: disable=too-many-function-args

        X = dataset.X[self.columns]
        estimator.fit(X, dataset.y)
        self.estimator = estimator

        result = permutation_importance(
            estimator,
            X,
            dataset.y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.get_config('random_state'),
            n_jobs=1
        )

        importances = np.asarray(result.importances_mean)
        stds = np.asarray(result.importances_std)

        self.importances = {
            column: float(value)
            for column, value in zip(self.columns, importances)
        }
        if stds.size:
            self.importances_std = {
                column: float(value)
                for column, value in zip(self.columns, stds)
            }

        threshold_value = self._resolve_threshold(importances)
        if threshold_value is not None and not np.isfinite(threshold_value):
            threshold_value = None
        self.threshold_value = threshold_value

        if threshold_value is None:
            candidate_mask = np.ones(len(self.columns), dtype=bool)
        else:
            candidate_mask = np.isfinite(importances) & (importances > threshold_value)

        candidate_indices = np.where(candidate_mask)[0].tolist()
        selected_indices = candidate_indices

        if max_features is not None:
            ranking = np.where(np.isfinite(importances), importances, -np.inf)
            order = np.argsort(ranking)[::-1]
            order = [idx for idx in order if candidate_mask[idx]]
            selected_indices = list(order[:max_features])

        selected_indices = sorted(selected_indices)
        self.selected_columns = list(pd.Index(self.columns)[selected_indices])
        selected_set = set(self.selected_columns)
        self.columns_to_drop = [col for col in self.columns if col not in selected_set]

        if self.columns_to_drop:
            threshold_display = None
            threshold_config = self.get_config('threshold')
            if threshold_value is not None:
                if isinstance(threshold_config, str) and \
                        threshold_config.strip().lower() in ['mean', 'median']:
                    threshold_display = f"{threshold_config} ({threshold_value:.6g})"
                else:
                    threshold_display = f"{threshold_value:.6g}"

            top_limit = max_features
            top_set = set(self.selected_columns)

            for column in self.columns_to_drop:
                importance = self.importances.get(column)
                if importance is None or not np.isfinite(importance):
                    if threshold_value is not None:
                        message = (
                            f"Dropped column **`{column}`** because its permutation "
                            "importance was not finite."
                        )
                    elif top_limit is not None:
                        message = (
                            f"Dropped column **`{column}`** because it was not in "
                            f"the top **{top_limit}** features by permutation importance."
                        )
                    else:
                        message = (
                            f"Dropped column **`{column}`** because it was not selected "
                            "by permutation importance."
                        )
                else:
                    if threshold_value is not None and importance <= threshold_value:
                        message = (
                            f"Dropped column **`{column}`** because its permutation "
                            f"importance ({importance:.6g}) was below the threshold "
                            f"({threshold_display})."
                        )
                    elif top_limit is not None and column not in top_set:
                        message = (
                            f"Dropped column **`{column}`** because it was not in "
                            f"the top **{top_limit}** features by permutation importance "
                            f"({importance:.6g})."
                        )
                    else:
                        message = (
                            f"Dropped column **`{column}`** because it was not selected "
                            f"by permutation importance ({importance:.6g})."
                        )
                self.explanations.append(message)

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

        if self._build_estimator(dataset) is None:
            return False

        if self._resolve_n_repeats() is None:
            return False

        if not self._threshold_config_valid():
            return False

        threshold = self.get_config('threshold')
        max_features = self._resolve_max_features(len(columns))
        if (threshold is None or (
            isinstance(threshold, str)
            and threshold.strip().lower() in ['none', '']
        )) and max_features is None:
            return False

        return True

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
