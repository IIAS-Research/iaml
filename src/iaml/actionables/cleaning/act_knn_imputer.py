"""[STEP] KNN imputer for numeric columns."""
import textwrap

import pandas as pd
from sklearn.impute import KNNImputer

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('cleaning')
class ActKNNImputer(Actionable):
    """[STEP] Impute missing numeric values using KNN."""

    name: str = 'Impute missing values (KNN)'
    _usage: str = 'Use when numeric features have gaps and you want to preserve rows vs ActDropNumericalColumn. Applicable to numeric columns with enough non-missing rows; use ActCategoricalImputer for categoricals. Avoid when missingness is extreme, dataset is tiny, or you plan to drop the column.'
    _description: str = textwrap.dedent('''\
        Impute missing numeric values using a k-nearest neighbors strategy.''')
    _description_long: str = textwrap.dedent('''\
        Uses scikit-learn KNNImputer to fill missing values in numeric columns
        by averaging the k nearest neighbors in feature space. Non-numeric
        columns are left untouched.''')

    def __init__(self):
        self.columns: list[str] = []
        self.knn_columns: list[str] = []
        self.imputer: KNNImputer | None = None
        self._all_nan_cols: list[str] = []
        self._fallback_values: dict[str, float] = {}
        self._nan_stats: dict[str, tuple[int, int, float]] = {}

        self.configuration = {
            'n_neighbors': {
                'description': 'Number of neighbors used for imputing missing values.',
                'default': 5,
                'range': [1, 50],
                'passthrough': False
            },
            'weights': {
                'description': 'Weight function used in prediction.',
                'default': 'uniform',
                'categorical': ['uniform', 'distance']
            },
            'metric': {
                'description': 'Distance metric to use for missing-aware KNN.',
                'default': 'nan_euclidean',
                'categorical': ['nan_euclidean']
            }
        }

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.knn_columns = []
        self.imputer = None
        self._all_nan_cols = []
        self._fallback_values = {}
        self._nan_stats = {}
        self.explanations = []

        if not self.columns:
            return self

        X_num = dataset.X[self.columns]
        total_rows = len(X_num)
        if total_rows == 0:
            self.explanations = ["Skipped KNN imputation: dataset has 0 rows."]
            return self

        missing_counts = X_num.isna().sum()
        means = X_num.mean()
        for column in self.columns:
            missing = int(missing_counts[column])
            pct = (missing / total_rows * 100.0) if total_rows else 0.0
            self._nan_stats[column] = (missing, total_rows, pct)

            mean_value = means[column]
            if pd.isna(mean_value):
                mean_value = 0.0
            self._fallback_values[column] = float(mean_value)

        self._all_nan_cols = [
            column for column in self.columns if missing_counts[column] == total_rows
        ]
        self.knn_columns = [
            column for column in self.columns if column not in self._all_nan_cols
        ]

        if self.knn_columns and total_rows >= 2:
            n_neighbors = min(self.get_config('n_neighbors'), total_rows - 1)
            n_neighbors = max(1, int(n_neighbors))
            params = self.passthrough_parameters()
            params['n_neighbors'] = n_neighbors
            self.imputer = KNNImputer(**params)
            self.imputer.fit(X_num[self.knn_columns])

        if self.imputer is not None:
            self.explanations = [
                f"Imputed missing values of column **`{c}`** using **KNN** "
                f"(**{n}** / **{t}**; **{pct:.2f}%** missing in train data)."
                for c, (n, t, pct) in self._nan_stats.items()
                if n > 0 and c in self.knn_columns
            ]
        if self._all_nan_cols:
            self.explanations.append(
                "Filled all-NaN numeric columns with 0.0: " +
                ", ".join(f"`{c}`" for c in self._all_nan_cols) +
                "."
            )
        if self.imputer is None and not self.explanations:
            if any(n > 0 for n, _, _ in self._nan_stats.values()):
                self.explanations.append(
                    "Skipped KNN imputation; filled numeric columns with their mean "
                    "(0.0 when undefined)."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            return X

        if self.imputer is not None and self.knn_columns:
            if all(column in X.columns for column in self.knn_columns):
                X_knn = X[self.knn_columns].copy()
                X_knn = X_knn.apply(pd.to_numeric, errors='coerce')
                imputed = self.imputer.transform(X_knn)
                X.loc[:, self.knn_columns] = imputed

        for column, fill_value in self._fallback_values.items():
            if column in X.columns:
                X[column] = X[column].fillna(fill_value).infer_objects(copy=False)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        return bool(dataset.X[columns].isna().any().any())

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None or candidate.dataset.X.empty:
            return 0.0
        columns = candidate.dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns:
            return 0.0
        missing = candidate.dataset.X[columns].isna().sum().sum()
        total = candidate.dataset.X[columns].size or 1
        return min(1.0, missing / total)
