"""[STEP] Add KMeans distance/cluster features."""
import textwrap
import pandas as pd
from sklearn.cluster import KMeans
from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActKMeansFeatures(Actionable):
    """[STEP] Add KMeans distance/cluster features."""

    name: str = "KMeans Features"
    _description: str = "Add KMeans distance and cluster assignment features"
    _usage: str = "Use when numeric features may form clusters and you want distance/cluster signals (vs ActKernelPCA or ActFeatureAgglomeration). Applicable to numeric tables with enough rows for k clusters. Avoid when data are tiny, mostly categorical, or clustering adds noise."
    _description_long: str = textwrap.dedent('''\
        KMeans groups numeric observations into k clusters by minimizing the
        within-cluster variance. This step fits KMeans on numeric features and
        appends distance-to-centroid features and, optionally, the assigned
        cluster label. These additional features can surface non-linear structure
        in the numeric feature space for downstream models.
    ''')

    def __init__(self) -> None:
        self.configuration = {
            'n_clusters': {
                'description': 'Number of clusters to form.',
                'default': 8,
                'range': [2, 200]
            },
            'init': {
                'description': 'Initialization method.',
                'default': 'k-means++',
                'categorical': ['k-means++', 'random']
            },
            'n_init': {
                'description': 'Number of time the k-means algorithm will be run.',
                'default': 10,
                'range': [1, 20]
            },
            'max_iter': {
                'description': 'Maximum number of iterations per run.',
                'default': 300,
                'range': [50, 1000]
            },
            'tol': {
                'description': 'Relative tolerance with regards to inertia.',
                'default': 0.0001,
                'range': [1e-05, 0.01]
            },
            'algorithm': {
                'description': 'KMeans algorithm variant.',
                'default': 'lloyd',
                'categorical': ['lloyd', 'elkan']
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            },
            'add_distances': {
                'description': 'Whether to add distance-to-centroid features.',
                'default': True,
                'passthrough': False
            },
            'distance_mode': {
                'description': 'Distance features to add: all or closest.',
                'default': 'all',
                'categorical': ['all', 'closest'],
                'passthrough': False
            },
            'add_cluster_label': {
                'description': 'Whether to add the cluster assignment feature.',
                'default': True,
                'passthrough': False
            },
            'feature_prefix': {
                'description': 'Prefix for generated KMeans features.',
                'default': 'kmeans',
                'passthrough': False
            }
        }

        self.columns: list[str] = []
        self.active_columns: list[str] = []
        self.preprocessor: KMeans | None = None
        self.distance_feature_names: list[str] = []
        self.label_column: str | None = None
        self.feature_prefix: str = 'kmeans'
        self.add_distances: bool = True
        self.add_cluster_label: bool = True
        self.distance_mode: str = 'all'
        self.impute_values: pd.Series | None = None
        self.optimizable: bool = True

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ('1', 'true', 'yes', 'y'):
                return True
            if lowered in ('0', 'false', 'no', 'n'):
                return False
        return default

    @staticmethod
    def _coerce_str(value: object, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    @staticmethod
    def _resolve_distance_mode(value: object) -> str:
        if value is None:
            return 'all'
        text = str(value).strip().lower()
        if text in ('closest', 'min', 'nearest'):
            return 'closest'
        return 'all'

    @staticmethod
    def _select_active_columns(values: pd.DataFrame) -> list[str]:
        if values.empty:
            return []
        unique_counts = values.nunique(dropna=True)
        return [col for col in values.columns if unique_counts.get(col, 0) > 1]

    @staticmethod
    def _unique_name(name: str, reserved: set[str]) -> str:
        if name not in reserved:
            return name
        idx = 1
        candidate = f"{name}_{idx}"
        while candidate in reserved:
            idx += 1
            candidate = f"{name}_{idx}"
        return candidate

    def _resolve_n_clusters(self, n_samples: int) -> int | None:
        if n_samples < 2:
            return None
        n_clusters = self._coerce_int(self.get_config('n_clusters'), 8)
        n_clusters = max(2, n_clusters)
        return min(n_clusters, n_samples)

    def _build_kmeans(self, n_clusters: int) -> KMeans:
        params = self.passthrough_parameters()
        params['n_clusters'] = int(n_clusters)
        params['n_init'] = self._coerce_int(params.get('n_init'), 10)
        params['max_iter'] = self._coerce_int(params.get('max_iter'), 300)
        params['tol'] = float(params.get('tol', 0.0001))
        try:
            return KMeans(**params)
        except TypeError:
            params.pop('algorithm', None)
            return KMeans(**params)

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.active_columns = []
        self.preprocessor = None
        self.distance_feature_names = []
        self.label_column = None
        self.impute_values = None
        self.feature_prefix = self._coerce_str(self.get_config('feature_prefix'), 'kmeans')
        self.add_distances = self._coerce_bool(self.get_config('add_distances'), True)
        self.add_cluster_label = self._coerce_bool(self.get_config('add_cluster_label'), True)
        self.distance_mode = self._resolve_distance_mode(self.get_config('distance_mode'))
        self.explanations = []

        if not self.columns or dataset.X.empty:
            return self

        values = dataset.X[self.columns]
        self.active_columns = self._select_active_columns(values)
        if not self.active_columns:
            return self

        if not self.add_distances and not self.add_cluster_label:
            return self

        active_values = values[self.active_columns]
        n_clusters = self._resolve_n_clusters(active_values.shape[0])
        if n_clusters is None:
            return self

        self.configure('n_clusters', n_clusters) # pylint: disable=too-many-function-args
        self.impute_values = active_values.median(numeric_only=True)
        active_values = active_values.fillna(self.impute_values)

        self.preprocessor = self._build_kmeans(n_clusters)
        self.preprocessor.fit(active_values)

        reserved = set(dataset.X.columns)
        if self.add_distances:
            if self.distance_mode == 'closest':
                name = self._unique_name(f"{self.feature_prefix}_cluster_dist", reserved)
                self.distance_feature_names = [name]
                reserved.add(name)
            else:
                self.distance_feature_names = []
                for idx in range(n_clusters):
                    name = self._unique_name(f"{self.feature_prefix}_cluster_{idx}_dist", reserved)
                    self.distance_feature_names.append(name)
                    reserved.add(name)

        if self.add_cluster_label:
            self.label_column = self._unique_name(f"{self.feature_prefix}_cluster", reserved)
            reserved.add(self.label_column)

        if self.add_distances and self.distance_feature_names:
            self.explanations.append(
                f"Added {len(self.distance_feature_names)} KMeans distance features."
            )
        if self.add_cluster_label and self.label_column:
            self.explanations.append(
                f"Added KMeans cluster label feature `{self.label_column}`."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply KMeans distance/cluster features.

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None or not self.active_columns:
            return X

        if any(column not in X.columns for column in self.active_columns):
            return X

        values = X[self.active_columns]
        if self.impute_values is not None:
            values = values.fillna(self.impute_values)

        if self.add_distances and self.distance_feature_names:
            distances = self.preprocessor.transform(values)
            if self.distance_mode == 'closest':
                distances = distances.min(axis=1).reshape(-1, 1)
            distance_df = pd.DataFrame(
                distances,
                columns=self.distance_feature_names,
                index=X.index
            )
            X = pd.concat([X, distance_df], axis=1)

        if self.add_cluster_label and self.label_column:
            X[self.label_column] = self.preprocessor.predict(values)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        if not self._coerce_bool(self.get_config('add_distances'), True) \
            and not self._coerce_bool(self.get_config('add_cluster_label'), True):
            return False
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns:
            return False
        values = dataset.X[columns]
        active = self._select_active_columns(values)
        if not active:
            return False
        return self._resolve_n_clusters(values.shape[0]) is not None

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        dataset = candidate.dataset
        if not self.suitable(dataset):
            return 0.0
        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns:
            return 0.0
        values = dataset.X[columns]
        active = self._select_active_columns(values)
        if not active:
            return 0.0
        n_samples = max(1, values.shape[0])
        ratio = len(active) / n_samples
        return min(1.0, 0.2 + ratio)
