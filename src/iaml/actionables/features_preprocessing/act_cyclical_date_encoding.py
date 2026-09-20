"""Manual cyclical date encoding, available only through an explicit module import.

Date conversion and cleaning precede automatic feature preprocessing, so this
component needs an explicit position while datetime columns are still available.
"""
import textwrap
import numpy as np
import pandas as pd
from ...actionable import Actionable
from ...candidate import Candidate
from ...dataset import Dataset
from ...data_type import DataType
from ...decorators.all import is_step


@is_step('experimental')
class ActCyclicalDateEncoding(Actionable):
    """[STEP] Encode date columns with cyclical sine/cosine features."""

    name: str = "Cyclical Date Encoding"
    _description: str = "Encode date columns into sine/cosine features to capture cyclicity"
    _usage: str = "Use when datetime columns have cyclic parts (month/weekday/hour) and models need numeric features. Applicable to datetime columns with clear periodicity. Avoid when dates are non-cyclic or already encoded; consider ActKBinsDiscretizer or ActLogTransformer instead."
    _description_long: str = textwrap.dedent('''\
        Cyclical encoding turns calendar components (month, weekday, hour, etc.)
        into sine and cosine values. This preserves the circular nature of time,
        so end and start points on a cycle stay close in feature space while
        providing numeric values suitable for ML models.
    ''')

    _COMPONENTS: dict[str, tuple[str, int]] = {
        'month': ('month', 12),
        'dayofweek': ('dayofweek', 7),
        'day': ('day', 31),
        'hour': ('hour', 24),
        'minute': ('minute', 60),
        'second': ('second', 60),
    }
    _ALIASES: dict[str, str] = {
        'weekday': 'dayofweek',
        'dow': 'dayofweek',
        'dayofmonth': 'day',
        'dom': 'day',
    }

    def __init__(self) -> None:
        self.configuration = {
            'components': {
                'description': 'Date components to encode (month, dayofweek, day, hour, minute, second).',
                'default': ('month', 'dayofweek', 'day', 'hour')
            },
            'drop_original': {
                'description': 'Drop original date columns after encoding.',
                'default': True
            }
        }
        self.columns: list[str] = []
        self.encoding_plan: dict[str, list[tuple[str, str, int, str, str]]] = {}
        self.drop_original: bool = True

    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        return pd.api.types.is_datetime64_any_dtype(series)

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
    def _unique_name(name: str, reserved: set[str]) -> str:
        if name not in reserved:
            return name
        idx = 1
        candidate = f"{name}_{idx}"
        while candidate in reserved:
            idx += 1
            candidate = f"{name}_{idx}"
        return candidate

    def _resolve_components(self) -> list[str]:
        raw = self.get_config('components')
        if raw is None:
            raw_components = list(self._COMPONENTS.keys())
        elif isinstance(raw, str):
            raw_components = [raw]
        else:
            try:
                raw_components = list(raw)
            except TypeError:
                raw_components = [str(raw)]

        components: list[str] = []
        seen: set[str] = set()
        for component in raw_components:
            if component is None:
                continue
            component = str(component).strip().lower()
            if not component:
                continue
            component = self._ALIASES.get(component, component)
            if component not in self._COMPONENTS:
                continue
            if component in seen:
                continue
            seen.add(component)
            components.append(component)

        if not components:
            components = list(self._COMPONENTS.keys())

        return components

    def _build_plan(self, dataset: Dataset) -> dict[str, list[tuple[str, str, int, str, str]]]:
        columns = dataset.get_columns_names_by_type(DataType.DATE)
        if not columns or dataset.X.empty:
            return {}

        components = self._resolve_components()
        if not components:
            return {}

        reserved = set(dataset.X.columns)
        plan: dict[str, list[tuple[str, str, int, str, str]]] = {}

        for column in columns:
            if column not in dataset.X.columns:
                continue
            series = dataset.X[column]
            if not self._is_datetime(series):
                continue

            column_plan: list[tuple[str, str, int, str, str]] = []
            for component in components:
                accessor, period = self._COMPONENTS[component]
                values = getattr(series.dt, accessor)
                if values.nunique(dropna=True) <= 1:
                    continue
                base = f"{column}_{component}"
                sin_name = self._unique_name(f"{base}_sin", reserved)
                reserved.add(sin_name)
                cos_name = self._unique_name(f"{base}_cos", reserved)
                reserved.add(cos_name)
                column_plan.append((component, accessor, period, sin_name, cos_name))

            if column_plan:
                plan[column] = column_plan

        return plan

    def fit(self, dataset: Dataset) -> Actionable:
        self.encoding_plan = self._build_plan(dataset)
        self.columns = list(self.encoding_plan.keys())
        self.drop_original = self._coerce_bool(self.get_config('drop_original'), True)
        self.explanations = []

        for column, parts in self.encoding_plan.items():
            components = ", ".join([part[0] for part in parts])
            self.explanations.append(
                f'Encoded date column **`{column}`** into cyclical features ({components}).'
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cyclical encoding to date columns.

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if not self.encoding_plan:
            return X

        for column, parts in self.encoding_plan.items():
            if column not in X.columns:
                continue
            series = X[column]
            if not self._is_datetime(series):
                series = pd.to_datetime(series, errors='coerce')

            for _, accessor, period, sin_name, cos_name in parts:
                values = getattr(series.dt, accessor).astype(float)
                angles = (2.0 * np.pi * values) / float(period)
                X[sin_name] = np.sin(angles)
                X[cos_name] = np.cos(angles)

        if self.drop_original and self.columns:
            to_drop = [column for column in self.columns if column in X.columns]
            if to_drop:
                X = X.drop(columns=to_drop)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.X.empty:
            return False
        return bool(self._build_plan(dataset))

    def priorize(self, candidate: Candidate = None) -> float:
        if candidate is None:
            return 0.0
        dataset = candidate.dataset
        if dataset.X.empty:
            return 0.0
        return 0.5 if self._build_plan(dataset) else 0.0
