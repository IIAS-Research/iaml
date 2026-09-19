"""[STEP] Preprocess with QuantileTransformer"""
import textwrap
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActQuantileTransformer(Actionable):
    """[STEP] Preprocess with QuantileTransformer"""

    name: str = "Preprocess with QuantileTransformer"
    _description: str = textwrap.dedent('''\
        QuantileTransformer remaps numeric features to a uniform or normal distribution,
        reducing skew and making feature scales more comparable.''')
    _description_long: str = textwrap.dedent('''\
        QuantileTransformer estimates the empirical cumulative distribution for each
        numeric feature and maps values to a chosen target distribution.
        This non-linear transformation can reduce the impact of outliers and
        produce more Gaussian-like features for models that benefit from it.''')
    _usage: str = "Use when numeric features are skewed or unevenly scaled; compare ActKBinsDiscretizer. Applicable to continuous numeric columns before models preferring near-normal inputs. Avoid when original units or interpretability must stay, or data is very sparse; compare ActKernelPCA."

    def __init__(self):
        self.columns: list[str] = None
        self.preprocessor: QuantileTransformer = None

        self.configuration = {
            'n_quantiles': {
                'description': 'Number of quantiles to estimate.',
                'default': 1000,
                'range': [10, 1000]
            },
            'output_distribution': {
                'description': 'Target distribution for the transformed data.',
                'default': 'normal',
                'categorical': ['uniform', 'normal']
            },
            'subsample': {
                'description': 'Maximum number of samples used to estimate quantiles.',
                'default': 100000,
                'range': [1000, 200000]
            },
            'random_state': {
                'description': 'Random state used when subsampling.',
                'default': 42
            },
            'copy': {
                'description': 'Set to False to perform transformation in-place when possible.',
                'default': True,
                'categorical': [True, False]
            }
        }

        self.optimizable: bool = True

    def _build_transformer(self, n_samples: int) -> QuantileTransformer:
        params = self.passthrough_parameters()
        n_samples = max(1, int(n_samples))
        n_quantiles = int(params.pop('n_quantiles'))
        subsample = int(params.pop('subsample'))
        n_quantiles = max(1, min(n_quantiles, n_samples))
        subsample = max(1, min(subsample, n_samples))
        params['n_quantiles'] = n_quantiles
        params['subsample'] = subsample
        return QuantileTransformer(**params)

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns and not dataset.X.empty:
            values = dataset.X[self.columns]
            if values.isna().any().any():
                self.preprocessor = None
                return self
            self.preprocessor = self._build_transformer(values.shape[0])
            self.preprocessor.fit(values)
        else:
            self.preprocessor = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply QuantileTransformer

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor and self.columns:
            X[self.columns] = self.preprocessor.transform(X[self.columns])
        return X

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        values = dataset.X[columns]
        return not values.isna().any().any()

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        columns = candidate.dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or candidate.dataset.X.empty:
            return 0.0
        values = candidate.dataset.X[columns]
        if values.empty:
            return 0.0
        skewness = values.skew().abs().fillna(0.0)
        if skewness.empty:
            return 0.0
        mean_skew = float(skewness.mean())
        return min(1.0, mean_skew / 2.0)
