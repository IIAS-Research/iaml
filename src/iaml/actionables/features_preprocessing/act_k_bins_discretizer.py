"""[STEP] Discretize numeric features with KBinsDiscretizer"""
import textwrap
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActKBinsDiscretizer(Actionable):
    """[STEP] Discretize numeric features with KBinsDiscretizer"""

    name: str = "KBins Discretizer"
    _description: str = "Discretize numeric columns into uniform, quantile, or k-means bins"
    _usage: str = "Use when you want numeric features binned for simpler models or outlier robustness, rather than ActLogTransformer or ActKernelPCA. Applicable to continuous numeric columns with more than one unique value. Avoid when fine-grained ordering or exact values must be preserved."
    _description_long: str = textwrap.dedent('''\
        KBinsDiscretizer converts continuous numeric features into discrete bins.
        It can use uniform width bins, quantile-based bins, or k-means based bins.
        The resulting bins can be returned as ordinal values or one-hot encoded
        features, which may help models that prefer discrete inputs or benefit
        from reduced sensitivity to outliers.
    ''')

    def __init__(self):
        self.columns: list[str] = []
        self.active_columns: list[str] = []
        self.output_columns: list[str] = []
        self.preprocessor: KBinsDiscretizer | None = None

        self.configuration = {
            'n_bins': {
                'description': 'Number of bins to use for each numeric feature.',
                'default': 5,
                'range': [2, 50]
            },
            'encode': {
                'description': 'Encoding method for the transformed bins.',
                'default': 'ordinal',
                'categorical': ['ordinal', 'onehot', 'onehot-dense']
            },
            'strategy': {
                'description': 'Strategy used to define the widths of the bins.',
                'default': 'quantile',
                'categorical': ['uniform', 'quantile', 'kmeans']
            },
            'subsample': {
                'description': 'Maximum number of samples used to estimate quantile bin edges.',
                'default': 200000,
                'range': [1000, 500000]
            },
            'random_state': {
                'description': 'Random state used when strategy is kmeans.',
                'default': 42
            }
        }

        self.optimizable: bool = True

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _select_active_columns(values: pd.DataFrame) -> list[str]:
        if values.empty:
            return []
        unique_counts = values.nunique(dropna=True)
        return [col for col in values.columns if unique_counts.get(col, 0) > 1]

    @staticmethod
    def _resolve_n_bins(values: pd.DataFrame, base_bins: int) -> int | list[int]:
        if values.empty:
            return base_bins
        unique_counts = values.nunique(dropna=True).astype(int)
        per_feature_bins = [min(base_bins, max(2, count)) for count in unique_counts]
        if not per_feature_bins:
            return base_bins
        if all(bins == per_feature_bins[0] for bins in per_feature_bins):
            return per_feature_bins[0]
        return per_feature_bins

    def _feature_names(self) -> list[str]:
        if self.output_columns:
            return self.output_columns
        if self.preprocessor is None:
            return []
        if hasattr(self.preprocessor, "get_feature_names_out"):
            try:
                return list(self.preprocessor.get_feature_names_out(self.active_columns))
            except ValueError:
                return []
        if hasattr(self.preprocessor, "n_bins_"):
            names: list[str] = []
            for column, n_bins in zip(self.active_columns, self.preprocessor.n_bins_):
                for idx in range(int(n_bins)):
                    names.append(f"{column}_bin_{idx}")
            return names
        return []

    def _build_transformer(self, n_samples: int, n_bins: int | list[int]) -> KBinsDiscretizer:
        params = self.passthrough_parameters()
        params['n_bins'] = n_bins
        if 'subsample' in params:
            subsample = self._coerce_int(params.get('subsample'), n_samples)
            subsample = max(1, min(subsample, n_samples))
            params['subsample'] = subsample
        return KBinsDiscretizer(**params)

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.active_columns = []
        self.output_columns = []
        self.preprocessor = None

        if not self.columns or dataset.X.empty:
            return self

        values = dataset.X[self.columns]
        self.active_columns = self._select_active_columns(values)
        if not self.active_columns:
            return self

        active_values = values[self.active_columns]
        base_bins = self._coerce_int(self.get_config('n_bins'), 5)
        base_bins = max(2, base_bins)
        n_bins = self._resolve_n_bins(active_values, base_bins)

        self.preprocessor = self._build_transformer(active_values.shape[0], n_bins)
        self.preprocessor.fit(active_values)

        if self.get_config('encode') in ('onehot', 'onehot-dense'):
            self.output_columns = self._feature_names()
        else:
            self.output_columns = self.active_columns.copy()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply KBinsDiscretizer

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None or not self.active_columns:
            return X

        values = X[self.active_columns]
        transformed = self.preprocessor.transform(values)
        encode = self.get_config('encode')

        if encode == 'ordinal':
            X[self.active_columns] = transformed
            return X

        X = X.reset_index(drop=True)
        feature_names = self.output_columns or self._feature_names()

        if hasattr(transformed, "toarray") and encode == 'onehot':
            encoded_df = pd.DataFrame.sparse.from_spmatrix(
                transformed,
                columns=feature_names,
                index=X.index
            )
        else:
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            encoded_df = pd.DataFrame(transformed, columns=feature_names, index=X.index)

        X = X.drop(columns=self.active_columns)
        return pd.concat([X, encoded_df], axis=1)

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        values = dataset.X[columns]
        if values.empty:
            return False
        unique_counts = values.nunique(dropna=True)
        return bool((unique_counts > 1).any())

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        dataset = candidate.dataset
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return 0.0
        values = dataset.X[columns]
        if values.empty:
            return 0.0
        n_rows = max(1, values.shape[0])
        unique_ratio = values.nunique(dropna=True) / n_rows
        if unique_ratio.empty:
            return 0.0
        mean_ratio = float(unique_ratio.mean())
        return min(1.0, mean_ratio * 2.0)
