"""[STEP] Impute missing values with MICE (miceforest/LightGBM)"""
import textwrap
import pandas as pd
import numpy as np
import types

from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType

from ...logger import Logger

# miceforest
try:
    import miceforest as mf
except ImportError as e:
    raise ImportError(
        "miceforest is required for ActMICEForestImputer. Install with: pip install miceforest lightgbm"
    ) from e


@is_step('cleaning')
class ActMICEForestImputer(Actionable):
    """[STEP] Impute missing values with MICE (miceforest/LightGBM)."""

    name: str = 'Impute missing values (MICE - miceforest)'
    _description: str = textwrap.dedent('''\
        Impute missing values using MICE (chained equations) powered by LightGBM,
        leveraging multivariate relations between features (numeric only by default).''')
    _description_long: str = textwrap.dedent('''\
        Uses miceforest.ImputationKernel to iteratively impute missing values.
        By default works on numeric columns. Optionally auto-categorizes low-cardinality
        object columns to allow categorical imputation by LightGBM.''')
    can_be_disabled: bool = False

    def __init__(self):
        self.columns: list[str] = None
        self.kernel: mf.ImputationKernel | None = None
        self._nan_stats: dict[str, tuple[int, int, float]] = {}
        self._all_nan_cols: list[str] = []
        self._prefill_values: dict[str, object] = {}

        self.configuration: dict = {
            'max_iter': {
                'description': 'Number of MICE iterations to run.',
                'default': 5
            },
            'random_state': {
                'description': 'Random seed for reproducibility (None for stochastic).',
                'default': 0
            },
            'auto_categorize': {
                'description': 'If True, cast low-cardinality object columns to category.',
                'default': False
            },
            'auto_categorize_max_cardinality': {
                'description': 'Max unique values to auto-cast object->category when auto_categorize=True.',
                'default': 30
            }
        }

    # --- helpers -----------------------------------------------------------------
    def _select_columns(self, df: pd.DataFrame) -> list[str]:
        # Base : colonnes numériques
        cols = list(df.columns.intersection(df.select_dtypes(include=[np.number]).columns))

        if self.configuration['auto_categorize']['default']:
            max_card = int(self.configuration['auto_categorize_max_cardinality']['default'])
            obj_cols = df.select_dtypes(include=['object']).columns
            for c in obj_cols:
                nuniq = df[c].nunique(dropna=True)
                if 1 < nuniq <= max_card:
                    df[c] = df[c].astype('category')
                    cols.append(c)
            cat_cols = df.select_dtypes(include=['category']).columns
            for c in cat_cols:
                if c not in cols:
                    cols.append(c)

        # Préserver l'ordre d’origine
        cols_ordered = [c for c in df.columns if c in set(cols)]
        return cols_ordered

    # --- core API ----------------------------------------------------------------
    def fit(self, dataset: Dataset) -> Actionable:
        X = dataset.X.copy()
        self.columns = self._select_columns(X)
        if not self.columns:
            self.explanations = []
            self.kernel = None
            return self
        
        Logger().info("MICE FIT")
        

        X_fit = X[self.columns].copy()

        # Exclure colonnes entièrement NaN (miceforest ne peut pas les initialiser)
        nonnull_counts = X_fit.notna().sum(axis=0)
        valid_cols = [c for c in self.columns if nonnull_counts[c] > 0]
        self._all_nan_cols = [c for c in self.columns if nonnull_counts[c] == 0]

        # Pré-remplissages par défaut pour ces colonnes (appliqués en transform)
        self._prefill_values = {}
        for c in self._all_nan_cols:
            if pd.api.types.is_numeric_dtype(X_fit[c]):
                self._prefill_values[c] = 0
            elif pd.api.types.is_categorical_dtype(X_fit[c]):
                if 'missing' not in X_fit[c].cat.categories:
                    X_fit[c] = X_fit[c].cat.add_categories(['missing'])
                self._prefill_values[c] = 'missing'
            else:
                self._prefill_values[c] = 'missing'

        if not valid_cols:
            self.explanations = [
                "Skipped MICE: all selected columns had 0 observed values."
            ]
            self.kernel = None
            return self

        X_fit_valid = X_fit[valid_cols].copy().reset_index(drop=True)

        self._nan_stats = {}
        for c in valid_cols:
            n_missing = int(X_fit_valid[c].isna().sum())
            n_total = int(len(X_fit_valid[c]))
            pct = (n_missing / n_total * 100.0) if n_total > 0 else 0.0
            self._nan_stats[c] = (n_missing, n_total, pct)
        
        X_fit_valid = X_fit[valid_cols].copy().reset_index(drop=True)
        rs = self.configuration['random_state']['default']

        self.kernel = mf.ImputationKernel(
            data=X_fit_valid,
            random_state=rs,
            num_datasets=1,
            save_all_iterations_data=True,
        )
        
        self.kernel.mice(
            int(self.configuration['max_iter']['default']),
            n_jobs=8,
            verbose=False,
            seed=rs,
            random_state=rs)
        
        self._ensure_seed_on_kernel_models(rs)

        # Explications
        expl = [
            f"Imputed missing values of column **`{c}`** using **MICE (miceforest)** "
            f"(**{n}** / **{t}**; **{pct:.2f}%** missing in train data)."
            for c, (n, t, pct) in self._nan_stats.items() if n > 0
        ]
        if self._all_nan_cols:
            expl.append(
                "Skipped MICE for all-NaN columns: " +
                ", ".join(f"`{c}`" for c in self._all_nan_cols) +
                " (cannot initialize with miceforest)."
            )
        self.explanations = expl

        # Conserver toutes les colonnes (ordre : valides puis all-NaN)
        self.columns = valid_cols + self._all_nan_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply trained MICEForest kernel to new data."""
        if not self.columns:
            return X

        Logger().info("MICE TRANSFORM")

        X_out = X.copy()

        # Harmoniser les types si auto_categorize activé
        if self.configuration['auto_categorize']['default']:
            max_card = int(self.configuration['auto_categorize_max_cardinality']['default'])
            for c in X_out.columns:
                if c in self.columns and X_out[c].dtype == 'object':
                    nuniq = X_out[c].nunique(dropna=True)
                    if 1 < nuniq <= max_card:
                        X_out[c] = X_out[c].astype('category')

        # 1) Imputer les colonnes "valides" via MICE (celles non all-NaN au fit)
        valid_cols = [c for c in self.columns if c not in self._all_nan_cols]
        if self.kernel is not None and valid_cols:
            X_sub = X_out[valid_cols].copy()

            # miceforest attend un RangeIndex
            X_sub_reset = X_sub.reset_index(drop=True)

            # Sécuriser les modèles du kernel : s'assurer que params['seed'] existe
            _ = self._ensure_seed_on_kernel_models(self.configuration['random_state']['default'])

            # Appel principal à impute_new_data ; en cas de KeyError 'seed', on coupe le PMM
            try:
                imputed_data = self.kernel.impute_new_data(
                    new_data=X_sub_reset,
                    datasets=[0],
                    iterations=int(self.configuration['max_iter']['default'])
                )
            except KeyError as e:
                if str(e) == "'seed'":
                    # Fallback : désactiver le PMM pour l'imputation "new data"
                    imputed_data = self.kernel.impute_new_data(
                        new_data=X_sub_reset,
                        datasets=[0],
                        iterations=int(self.configuration['max_iter']['default']),
                        mean_match_candidates=0  # imputation par prédiction directe
                        # exact=True  # <- alternative possible selon versions de miceforest
                    )
                else:
                    raise

            imputed = imputed_data.complete_data(dataset=0)

            # Réinjection en respectant l'index d'origine de X_out
            X_out.loc[:, valid_cols] = imputed[valid_cols].values

        # 2) Pré-remplir les colonnes all-NaN (impossibles à traiter par miceforest)
        for c in self._all_nan_cols:
            if c in X_out.columns:
                fill_val = self._prefill_values.get(c, np.nan)
                if pd.api.types.is_categorical_dtype(X_out[c]) and str(fill_val) not in X_out[c].cat.categories:
                    X_out[c] = X_out[c].cat.add_categories([fill_val])
                X_out[c] = X_out[c].fillna(fill_val)

        # 3) Nettoyer les types numériques
        for col in X_out.columns:
            if pd.api.types.is_numeric_dtype(X_out[col]):
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').infer_objects(copy=False)

        return X_out


    def priorize(self, candidate: Candidate = None) -> float:
        alpha = 0.01
        return 1 - (candidate.dataset.X.isnull().sum().min() / len(candidate.dataset.X)) + alpha
    
    def _yield_kernel_models(self):
        """Itère de manière sécurisée sur les modèles LGBM stockés dans le kernel miceforest,
        sans introspection profonde qui casse sur pandas."""
        if self.kernel is None:
            return
        containers = []
        for attr in ("imputation_models", "models", "model_dict", "model_dicts"):
            if hasattr(self.kernel, attr):
                containers.append(getattr(self.kernel, attr))

        def walk(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from walk(v)
            else:
                # Cible : objets LightGBM-like avec un dict .params
                p = getattr(obj, "params", None)
                if isinstance(p, dict):
                    yield obj
                    
        for c in containers:
            yield from walk(c)

    def _ensure_seed_on_kernel_models(self, seed: int) -> int:
        """Ajoute params['seed'] aux modèles si absent. Retourne le nombre patché."""
        patched = 0
        for m in self._yield_kernel_models():
            p = getattr(m, "params", None)
            if isinstance(p, dict) and "seed" not in p:
                p["seed"] = p.get("random_state", int(seed) if seed is not None else 0)
                patched += 1
        return patched
