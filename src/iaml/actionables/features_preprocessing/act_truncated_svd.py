"""[STEP] Reduce dimensions with TruncatedSVD"""
import textwrap
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActTruncatedSVD(Actionable):
    """[STEP] Reduce dimensions with TruncatedSVD"""

    name: str = "TruncatedSVD"
    _description: str = "Reduce dimensionality for sparse or high-dimensional numeric features"
    _description_long: str = textwrap.dedent('''\
        TruncatedSVD performs a low-rank approximation of the feature matrix.
        It is well suited for sparse representations such as TF-IDF or hashing
        vectors, and can reduce the number of features while preserving most
        of the structure of the data.
    ''')
    _usage: str = "Use when you need linear reduction for sparse, high-dimensional numeric features; compare ActKernelPCA for nonlinear patterns. Applicable to TF-IDF, hashing, and large numeric feature matrices. Avoid when data is dense with nonlinear structure or when ActFastICA is the goal."

    def __init__(self):
        self.columns: list[str] = []
        self.preprocessor: TruncatedSVD | None = None
        self.component_names: list[str] = []

        self.configuration = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [2, 2000]
            },
            'algorithm': {
                'description': 'SVD solver to use.',
                'default': 'randomized',
                'categorical': ['randomized', 'arpack']
            },
            'n_iter': {
                'description': 'Number of power iterations for randomized SVD.',
                'default': 5,
                'range': [2, 15]
            },
            'tol': {
                'description': 'Tolerance for arpack solver.',
                'default': 0.0,
                'range': [0.0, 0.1]
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

    def _resolve_n_components(self, n_samples: int, n_features: int) -> int | None:
        max_components = min(n_samples - 1, n_features - 1)
        if max_components < 1:
            return None
        n_components = self._coerce_int(self.get_config('n_components'), max_components)
        n_components = max(1, n_components)
        return min(n_components, max_components)

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.preprocessor = None
        self.component_names = []

        if not self.columns or dataset.X.empty:
            return self

        values = dataset.X[self.columns]
        if values.isna().any().any():
            return self
        n_components = self._resolve_n_components(*values.shape)
        if n_components is None:
            return self

        self.configure('n_components', n_components) # pylint: disable=too-many-function-args
        params = self.passthrough_parameters()
        self.preprocessor = TruncatedSVD(**params)
        self.preprocessor.fit(values)

        self.component_names = [f"svd_{i}" for i in range(n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply TruncatedSVD

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None or not self.columns:
            return X

        values = X[self.columns]
        transformed = self.preprocessor.transform(values)
        component_names = self.component_names or [
            f"svd_{i}" for i in range(transformed.shape[1])
        ]
        svd_df = pd.DataFrame(transformed, columns=component_names, index=X.index)

        X = X.drop(columns=self.columns)
        return pd.concat([X, svd_df], axis=1)

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
        n_features = len(columns)
        n_samples = dataset.X.shape[0]
        if n_features <= 1 or n_samples <= 1:
            return 0.0
        return min(1.0, n_features / max(1, n_samples))
