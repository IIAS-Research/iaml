"""[STEP] Reduce dimensions with FastICA"""
import textwrap
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.utils.validation import check_array
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActFastICA(Actionable):
    """[STEP] Reduce dimensions with FastICA"""

    name: str = "FastICA"
    _description: str = "Apply FastICA to extract independent components from numeric features"
    _description_long: str = textwrap.dedent('''\
        FastICA is a linear decomposition technique that separates mixed signals
        into statistically independent components. It can be used as a
        dimensionality reduction step by projecting numeric features into a
        smaller set of latent sources while keeping the transformation
        deterministic and efficient.
    ''')
    _usage: str = "Use when linear independent components from mixed numeric signals are needed instead of ActKernelPCA. Applicable to continuous numeric features with enough samples. Avoid when samples or features are too few, or when ActFeatureAgglomeration is a better fit."

    def __init__(self):
        self.columns: list[str] = []
        self.component_names: list[str] = []
        self.preprocessor: FastICA | None = None

        self.configuration = {
            'n_components': {
                'description': 'Number of independent components to estimate.',
                'default': 10,
                'range': [2, 2000]
            },
            'algorithm': {
                'description': 'ICA algorithm to use.',
                'default': 'parallel',
                'categorical': ['parallel', 'deflation']
            },
            'whiten': {
                'description': 'Whether to whiten data before applying ICA.',
                'default': 'unit-variance',
                'categorical': ['unit-variance', 'arbitrary-variance', False]
            },
            'fun': {
                'description': 'Functional form of the G function.',
                'default': 'logcosh',
                'categorical': ['logcosh', 'exp', 'cube']
            },
            'max_iter': {
                'description': 'Maximum number of iterations during optimization.',
                'default': 200,
                'range': [100, 1000]
            },
            'tol': {
                'description': 'Convergence tolerance.',
                'default': 0.0001,
                'range': [1e-05, 0.01]
            },
            'random_state': {
                'description': 'Random State',
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

    def _resolve_n_components(self, values: pd.DataFrame) -> int | None:
        max_components = min(values.shape)
        if max_components >= 2 and self.get_config('whiten'):
            # Whitening can only use nonzero directions after centering.
            numeric = check_array(values, dtype=[np.float64, np.float32])
            max_components = int(np.linalg.matrix_rank(numeric - numeric.mean(axis=0)))
        if max_components < 2:
            return None
        n_components = self._coerce_int(self.get_config('n_components'), max_components)
        n_components = max(2, n_components)
        return min(n_components, max_components)

    def _build_transformer(self) -> FastICA:
        params = self.passthrough_parameters()
        params['n_components'] = int(params['n_components'])
        params['max_iter'] = int(params['max_iter'])
        params['tol'] = float(params['tol'])
        return FastICA(**params)

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.preprocessor = None
        self.component_names = []

        if not self.columns or dataset.X.empty:
            return self

        values = dataset.X[self.columns]
        if values.isna().any().any():
            return self
        n_components = self._resolve_n_components(values)
        if n_components is None:
            return self

        self.configure('n_components', n_components) # pylint: disable=too-many-function-args
        self.preprocessor = self._build_transformer()
        self.preprocessor.fit(values)

        self.component_names = [f"ica_{i}" for i in range(n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply FastICA

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None or not self.columns:
            return X

        values = X[self.columns]
        transformed = self.preprocessor.transform(values)
        component_names = self.component_names or [
            f"ica_{i}" for i in range(transformed.shape[1])
        ]
        ica_df = pd.DataFrame(transformed, columns=component_names, index=X.index)

        X = X.drop(columns=self.columns)
        return pd.concat([X, ica_df], axis=1)

    def suitable(self, dataset: Dataset) -> bool:
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        values = dataset.X[columns]
        if values.isna().any().any():
            return False
        return min(values.shape) > 1

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        dataset = candidate.dataset
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return 0.0
        values = dataset.X[columns]
        if min(values.shape) <= 1:
            return 0.0
        n_features = values.shape[1]
        n_samples = values.shape[0]
        return min(1.0, n_features / max(1, n_samples))
