"""[STEP] Target mean encoding with internal cross-validation."""
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActTargetEncoder(Actionable):
    """[STEP] Encode categorical features by target mean with internal CV."""

    name: str = 'Target mean encoding'
    _description: str = textwrap.dedent('''\
        Encode categorical columns with the mean of the target using internal CV.''')
    _description_long: str = textwrap.dedent('''\
        Replace each category with the mean target value computed on training folds only.
        This yields out-of-fold encodings for the training data to reduce leakage. For new
        data, mappings learned on the full training set are used and unseen categories are
        mapped to the global target mean. Non-numeric targets are factorized first.''')
    _usage: str = "Use when categorical features have target signal and you want numeric encoding vs ActCountVectorizer. Applicable to supervised data with categorical columns and enough rows. Avoid when leakage risk is high or categories are too sparse; prefer ActDropHighCardinalityCategorical."

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.encodings: dict[str, dict[Any, float]] = {}
        self.fallback_values: dict[str, float] = {}
        self.global_mean: float = 0.0
        self._train_encoded: pd.DataFrame | None = None
        self._train_X: pd.DataFrame | None = None

        self.configuration = {
            'n_splits': {
                'description': 'Number of CV folds used to compute out-of-fold encodings.',
                'default': 5
            },
            'shuffle': {
                'description': 'Shuffle rows before splitting into folds.',
                'default': True
            },
            'random_state': {
                'description': 'Random seed used when shuffling.',
                'default': 42
            },
            'smoothing': {
                'description': 'Smoothing strength towards the global mean.',
                'default': 1.0
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = self._select_columns(dataset)
        self.encodings = {}
        self.fallback_values = {}
        self.global_mean = 0.0
        self._train_encoded = None
        self._train_X = None
        self.explanations = []

        if not self.columns or dataset.X.empty or dataset.y is None:
            return self

        y_series = pd.Series(dataset.y, index=dataset.X.index)
        y_numeric = self._coerce_target(y_series)
        if y_numeric.empty:
            return self

        prior = float(y_numeric.mean())
        if np.isnan(prior):
            prior = 0.0
        self.global_mean = prior

        n_rows = len(dataset.X)
        n_splits = self._coerce_int(self.get_config('n_splits'), 5)
        n_splits = min(max(n_splits, 2), n_rows) if n_rows > 1 else 1
        shuffle = bool(self.get_config('shuffle'))
        random_state = self._coerce_int(self.get_config('random_state'), 42)
        smoothing = self._coerce_float(self.get_config('smoothing'), 1.0)

        use_cv = n_rows >= 2 and n_splits >= 2
        splits = []
        if use_cv:
            splits = self._build_splits(
                dataset.X,
                y_series,
                dataset.type_of_target,
                n_splits,
                shuffle,
                random_state
            )

        if use_cv and not splits:
            use_cv = False

        encoded_train = pd.DataFrame(index=dataset.X.index, columns=self.columns, dtype=float)

        for column in self.columns:
            series = dataset.X[column]
            mapping_full = self._fit_mapping(series, y_numeric, prior, smoothing)
            self.encodings[column] = mapping_full
            self.fallback_values[column] = prior

            if use_cv:
                oof = pd.Series(index=dataset.X.index, dtype=float)
                for train_idx, val_idx in splits:
                    mapping_fold = self._fit_mapping(
                        series.iloc[train_idx],
                        y_numeric.iloc[train_idx],
                        prior,
                        smoothing
                    )
                    encoded = series.iloc[val_idx].map(mapping_fold)
                    oof.iloc[val_idx] = encoded
                oof = oof.fillna(prior).astype(float)
            else:
                oof = series.map(mapping_full).fillna(prior).astype(float)

            encoded_train[column] = oof

            self.explanations.append(
                f"Target-encoded `{column}` using {len(mapping_full)} categories."
            )

        self._train_encoded = encoded_train
        self._train_X = dataset.X

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or X.empty:
            return X

        if self._train_X is not None and X is self._train_X and self._train_encoded is not None:
            for column in self.columns:
                if column in X.columns and column in self._train_encoded.columns:
                    X[column] = self._train_encoded[column].reindex(X.index).astype(float)
            return X

        for column in self.columns:
            if column not in X.columns:
                continue
            mapping = self.encodings.get(column, {})
            fallback = self.fallback_values.get(column, self.global_mean)
            fallback = self._coerce_float(fallback, self.global_mean)
            X[column] = X[column].map(mapping).fillna(fallback).astype(float)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.y is None or dataset.X.empty:
            return False
        if dataset.type_of_target == 'survival':
            return False
        return bool(self._select_columns(dataset))

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty or candidate.dataset.y is None:
            return 0.0

        columns = self._select_columns(candidate.dataset)
        if not columns:
            return 0.0

        total_rows = len(candidate.dataset.X)
        if total_rows <= 0:
            return 0.0

        unique_counts = candidate.dataset.X[columns].nunique(dropna=True)
        avg_cardinality = float((unique_counts / total_rows).mean())

        total_columns = candidate.dataset.X.shape[1] or 1
        cat_ratio = len(columns) / total_columns

        return min(1.0, max(cat_ratio, avg_cardinality))

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        category_columns = list(dataset.X.select_dtypes(include=['category']).columns)
        seen = set()
        ordered = []
        for column in columns + category_columns:
            if column in dataset.X.columns and column not in seen:
                ordered.append(column)
                seen.add(column)
        return ordered

    @staticmethod
    def _coerce_target(y: pd.Series) -> pd.Series:
        if y.empty:
            return y.astype(float)
        if pd.api.types.is_bool_dtype(y):
            return y.astype(float)
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(float)
        codes, _ = pd.factorize(y, sort=True)
        numeric = pd.Series(codes, index=y.index, dtype=float)
        numeric[codes < 0] = np.nan
        return numeric

    @staticmethod
    def _fit_mapping(
        series: pd.Series,
        target: pd.Series,
        prior: float,
        smoothing: float
    ) -> dict[Any, float]:
        if series.empty:
            return {}

        series_values = series.astype(object)
        grouped = target.groupby(series_values).agg(['mean', 'count'])
        if grouped.empty:
            return {}

        smoothing = float(smoothing)
        if smoothing <= 0:
            smooth = grouped['mean']
        else:
            smooth = (grouped['mean'] * grouped['count'] + prior * smoothing) \
                / (grouped['count'] + smoothing)

        return smooth.to_dict()

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_splits(
        X: pd.DataFrame,
        y: pd.Series,
        target_type: str,
        n_splits: int,
        shuffle: bool,
        random_state: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if n_splits < 2:
            return []

        is_classification = target_type not in (
            None,
            'continuous',
            'continuous-multioutput',
            'survival'
        )
        if is_classification:
            try:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=shuffle,
                    random_state=random_state
                )
                return list(splitter.split(X, y))
            except ValueError:
                pass

        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(X))
