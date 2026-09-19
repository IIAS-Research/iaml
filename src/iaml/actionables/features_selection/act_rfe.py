"""[STEP] Recursive Feature Elimination."""
import textwrap

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_selection')
class ActRFE(Actionable):
    """[STEP] Recursive Feature Elimination."""

    name: str = 'Recursive Feature Elimination'
    _usage: str = "Use when you need elimination to keep a set count of numeric features vs ActSelectKBest. Applicable to regression/classification with numeric inputs and linear/tree estimators. Avoid when feature count is huge or you need a fast filter; prefer ActRemoveLowVarianceColumn."
    _description: str = textwrap.dedent('''\
        Select {n_features_to_select} numeric features using RFE with
        a {estimator} estimator.''')
    _description_long: str = textwrap.dedent('''\
        Recursive Feature Elimination (RFE) trains a base estimator and
        removes the least important features at each iteration. This step
        supports linear and tree estimators to rank features and keep only
        the most relevant predictors.''')

    def __init__(self):
        self.configuration = {
            'estimator': {
                'description': 'Base estimator type used to rank features.',
                'default': 'linear',
                'categorical': ['linear', 'tree']
            },
            'n_features_to_select': {
                'description': 'Number of numeric features to keep.',
                'default': 10,
                'range': [1, 1000]
            },
            'step': {
                'description': 'Fraction of features removed at each iteration.',
                'default': 0.2,
                'range': [0.05, 1.0]
            },
            'random_state': {
                'description': 'Random seed for the underlying estimator.',
                'default': 42
            }
        }

        self.optimizable: bool = True
        self.columns: list[str] = []
        self.selected_columns: list[str] = []
        self.columns_to_drop: list[str] = []
        self.rankings: dict[str, int] = {}
        self.selector: RFE | None = None

    def _resolve_n_features_to_select(self, n_features: int) -> int:
        value = self.get_config('n_features_to_select')
        try:
            value = int(value)
        except (TypeError, ValueError):
            return 0

        if n_features <= 0:
            return 0

        return max(1, min(value, n_features))

    def _resolve_step(self) -> float | int | None:
        step_value = self.get_config('step')
        try:
            step_value = float(step_value)
        except (TypeError, ValueError):
            return None

        if step_value <= 0:
            return None

        if step_value >= 1:
            return int(step_value)

        return step_value

    def _build_estimator(self, dataset: Dataset):
        estimator_type = self.get_config('estimator')
        random_state = self.get_config('random_state')

        if dataset.type_of_target in ['continuous', 'continuous-multioutput']:
            if estimator_type == 'linear':
                return LinearRegression()
            if estimator_type == 'tree':
                return DecisionTreeRegressor(random_state=random_state)
            return None

        if dataset.type_of_target in ['binary', 'multiclass']:
            if estimator_type == 'linear':
                return LogisticRegression(
                    solver='liblinear',
                    max_iter=1000,
                    random_state=random_state
                )
            if estimator_type == 'tree':
                return DecisionTreeClassifier(random_state=random_state)
            return None

        return None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.selected_columns = []
        self.columns_to_drop = []
        self.rankings = {}
        self.selector = None
        self.explanations = []

        if dataset.y is None or dataset.type_of_target is None:
            return self

        if not self.columns:
            return self

        n_select = self._resolve_n_features_to_select(len(self.columns))
        if n_select < 1:
            return self

        if n_select >= len(self.columns):
            self.selected_columns = list(self.columns)
            return self

        estimator = self._build_estimator(dataset)
        if estimator is None:
            return self

        step_value = self._resolve_step()
        if step_value is None:
            return self

        if n_select != self.get_config('n_features_to_select'):
            self.configure('n_features_to_select', n_select)  # pylint: disable=too-many-function-args

        self.selector = RFE(
            estimator=estimator,
            n_features_to_select=n_select,
            step=step_value
        )
        self.selector.fit(dataset.X[self.columns], dataset.y)

        support = self.selector.get_support()
        self.selected_columns = list(pd.Index(self.columns)[support])
        self.columns_to_drop = list(pd.Index(self.columns)[~support])

        ranking = getattr(self.selector, 'ranking_', None)
        if ranking is not None:
            self.rankings = {
                column: int(rank) for column, rank in zip(self.columns, ranking)
            }

        for column in self.columns_to_drop:
            rank = self.rankings.get(column)
            if rank is None:
                self.explanations.append(
                    f"Dropped column **`{column}`** because it was not selected by RFE."
                )
            else:
                self.explanations.append(
                    f"Dropped column **`{column}`** because RFE ranked it **{rank}** "
                    f"(kept top **{n_select}**)."
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that were not selected.

        :param pd.DataFrame X: DataFrame to transform.
        :return: Transformed dataset.
        """
        if not self.columns_to_drop:
            return X

        drop_cols = [column for column in self.columns_to_drop if column in X.columns]
        if not drop_cols:
            return X
        return X.drop(columns=drop_cols)

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.y is None or dataset.type_of_target is None:
            return False

        if dataset.type_of_target == 'survival':
            return False

        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False

        n_select = self._resolve_n_features_to_select(len(columns))
        if n_select < 1 or n_select >= len(columns):
            return False

        if self._build_estimator(dataset) is None:
            return False

        if self._resolve_step() is None:
            return False

        return True

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
