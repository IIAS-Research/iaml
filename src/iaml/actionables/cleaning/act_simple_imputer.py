"""[STEP] Simple imputer dedicated to minimalist pipelines."""
import textwrap
from typing import Any

import pandas as pd

from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('minimal_preprocessing')
class ActSimpleImputer(Actionable):
    """[STEP] Lightweight imputer that fills NaNs with mean/mode values."""

    name: str = 'Simple Imputer'
    _description: str = textwrap.dedent('''\
        Fill missing numeric values with the column mean and categorical values with the
        most frequent value so minimalist predictors can run without preprocessing.''')
    _description_long: str = textwrap.dedent('''\
        This step mirrors a classical SimpleImputer: each numeric feature is imputed with
        its mean (fallback to 0 when undefined) while non-numeric columns are imputed with
        their most frequent value. It is primarily used to make minimalist predictors work
        on raw datasets that still contain NaNs.''')
    can_be_disabled: bool = False

    def __init__(self) -> None:
        self.numeric_fill_values: dict[str, float] = {}
        self.categorical_fill_values: dict[str, Any] = {}

    def fit(self, dataset: Dataset) -> 'ActSimpleImputer':
        X = dataset.X

        self.numeric_fill_values = {}
        self.categorical_fill_values = {}
        self.explanations = []

        numeric_cols = X.select_dtypes(include='number').columns
        for column in numeric_cols:
            series = X[column]
            missing = series.isna().sum()

            fill_value = series.mean()
            if pd.isna(fill_value):
                fill_value = 0.0

            self.numeric_fill_values[column] = float(fill_value)
            if missing > 0:
                self.explanations.append(
                    f"Filled {missing} numeric values in `{column}` with {fill_value:.4f}."
                )

        categorical_cols = X.select_dtypes(exclude='number').columns
        for column in categorical_cols:
            series = X[column]
            missing = series.isna().sum()

            mode_series = series.mode(dropna=True)
            if not mode_series.empty:
                fill_value = mode_series.iloc[0]
            elif pd.api.types.is_bool_dtype(series):
                fill_value = False
            else:
                fill_value = ''

            self.categorical_fill_values[column] = fill_value
            if missing > 0:
                self.explanations.append(
                    f"Filled {missing} categorical values in `{column}` with {fill_value!r}."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_fill_values:
            for column, fill_value in self.numeric_fill_values.items():
                if column in X.columns:
                    X[column] = X[column].fillna(fill_value)

        if self.categorical_fill_values:
            for column, fill_value in self.categorical_fill_values.items():
                if column in X.columns:
                    X[column] = X[column].fillna(fill_value)

        # As a last resort, replace any remaining NaNs to avoid downstream crashes.
        if X.isna().any().any():
            numeric_cols = X.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = X[numeric_cols].fillna(0.0)
            categorical_cols = X.select_dtypes(exclude='number').columns
            if len(categorical_cols) > 0:
                bool_cols = [c for c in categorical_cols if pd.api.types.is_bool_dtype(X[c])]
                other_cols = [c for c in categorical_cols if c not in bool_cols]
                if bool_cols:
                    X[bool_cols] = X[bool_cols].fillna(False)
                if other_cols:
                    X[other_cols] = X[other_cols].fillna('')

        return X

    def priorize(self, candidate: Candidate = None) -> float:  # pylint: disable=unused-argument
        if candidate is None or candidate.dataset.X.empty:
            return 0.0

        missing = candidate.dataset.X.isna().sum().sum()
        total = candidate.dataset.X.size or 1
        return min(1.0, missing / total)
