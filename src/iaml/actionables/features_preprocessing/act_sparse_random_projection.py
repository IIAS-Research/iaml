"""[STEP] Reduce dimensions with SparseRandomProjection."""
import textwrap
import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActSparseRandomProjection(Actionable):
    """[STEP] Reduce dimensions with SparseRandomProjection."""

    name: str = "SparseRandomProjection"
    _description: str = "Project numeric features to a lower-dimensional space quickly"
    _usage: str = "Use when you need fast, scalable reduction of many numeric features and can trade interpretability, vs heavier ActKernelPCA or ActFastICA. Applicable to high-dimensional numeric data, including sparse inputs. Avoid when features are few or you need interpretable axes."
    _description_long: str = textwrap.dedent('''\
        Sparse random projection compresses high-dimensional numeric features by
        multiplying them with a sparse random matrix. This preserves distances in
        expectation while keeping computation fast, making it suitable for large
        feature spaces where traditional decompositions are expensive.
    ''')

    def __init__(self):
        self.columns: list[str] = []
        self.component_names: list[str] = []
        self.preprocessor: SparseRandomProjection | None = None

        self.configuration = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [2, 2000]
            },
            'density': {
                'description': textwrap.dedent('''\
                    Proportion of non-zero elements in the projection matrix.
                    Use "auto" to rely on the default 1/sqrt(n_features).'''),
                'default': 'auto'
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
        max_components = min(n_samples, n_features)
        if max_components < 2:
            return None
        n_components = self._coerce_int(self.get_config('n_components'), max_components)
        n_components = max(2, n_components)
        return min(n_components, max_components)

    def _resolve_density(self) -> float | str:
        value = self.get_config('density')
        if value is None:
            return 'auto'
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in ['', 'auto', 'none']:
                return 'auto'
            try:
                value = float(stripped)
            except ValueError:
                return 'auto'
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 'auto'
        if numeric <= 0:
            return 'auto'
        return min(numeric, 1.0)

    def _build_transformer(self, n_components: int) -> SparseRandomProjection:
        params = self.passthrough_parameters()
        params['n_components'] = int(n_components)
        params['density'] = self._resolve_density()
        return SparseRandomProjection(**params)

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
        self.preprocessor = self._build_transformer(n_components)
        self.preprocessor.fit(values)

        self.component_names = [f"srp_{i}" for i in range(n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply SparseRandomProjection

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None or not self.columns:
            return X

        values = X[self.columns]
        transformed = self.preprocessor.transform(values)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        component_names = self.component_names or [
            f"srp_{i}" for i in range(transformed.shape[1])
        ]
        projected_df = pd.DataFrame(transformed, columns=component_names, index=X.index)

        X = X.drop(columns=self.columns)
        return pd.concat([X, projected_df], axis=1)

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
