"""[STEP] Select K Best Features."""
import textwrap

from collections.abc import Callable

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ...actionable import Actionable
from ...candidate import Candidate
from ...data_type import DataType
from ...dataset import Dataset
from ...decorators.all import is_step


@is_step('features_selection')
class ActSelectKBest(Actionable):
    """[STEP] Select K Best Features."""

    name: str = "Select K Best Features"
    _usage: str = "Use when you want a fast univariate top-k filter instead of ActRFE or ActSelectFromModel. Applicable to numeric features with a labeled target; for continuous targets use mutual_info. Avoid when you need multivariate interactions, stability, or chi2 with negative values."
    _description: str = textwrap.dedent('''\
        Select the {k} best numeric features using {score_func} scoring.''')
    _description_long: str = textwrap.dedent('''\
        SelectKBest ranks features based on a scoring function and keeps only
        the top k features. This step supports chi2, f_classif, and mutual_info
        scoring to reduce dimensionality and keep the most informative
        predictors for the target.''')

    def __init__(self):
        self.configuration = {
            'score_func': {
                'description': 'Score function used to rank features.',
                'default': 'f_classif',
                'categorical': ['chi2', 'f_classif', 'mutual_info']
            },
            'k': {
                'description': 'Number of top features to keep.',
                'default': 10,
                'range': [1, 1000]
            },
            'random_state': {
                'description': 'Random seed used for mutual_info scoring.',
                'default': 42
            }
        }

        self.optimizable: bool = True
        self.columns: list[str] = []
        self.selected_columns: list[str] = []
        self.columns_to_drop: list[str] = []
        self.selector: SelectKBest | None = None
        self._scores: dict[str, float] = {}

    def _resolve_score_func(self, dataset: Dataset) -> Callable | None:
        score_func = self.get_config('score_func')
        if score_func == 'chi2':
            return chi2
        if score_func == 'f_classif':
            return f_classif
        if score_func == 'mutual_info':
            random_state = self.get_config('random_state')
            if dataset.type_of_target == 'continuous':
                return lambda X, y: mutual_info_regression(
                    X, y, random_state=random_state
                )
            return lambda X, y: mutual_info_classif(
                X, y, random_state=random_state
            )
        return None

    def _resolve_k(self, max_features: int) -> int:
        k_value = self.get_config('k')
        try:
            k_value = int(k_value)
        except (TypeError, ValueError):
            return 0

        if max_features <= 0:
            return 0

        return max(1, min(k_value, max_features))

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.selected_columns = []
        self.columns_to_drop = []
        self.selector = None
        self._scores = {}
        self.explanations = []

        if not self.columns or dataset.y is None:
            return self
        if dataset.X[self.columns].isna().any().any():
            return self

        k_value = self._resolve_k(len(self.columns))
        if k_value < 1:
            return self

        score_func = self._resolve_score_func(dataset)
        if score_func is None:
            return self

        if k_value != self.get_config('k'):
            self.configure('k', k_value)  # pylint: disable=too-many-function-args

        self.selector = SelectKBest(score_func=score_func, k=k_value)
        self.selector.fit(dataset.X[self.columns], dataset.y)

        support = self.selector.get_support()
        self.selected_columns = list(pd.Index(self.columns)[support])
        self.columns_to_drop = list(pd.Index(self.columns)[~support])

        scores = self.selector.scores_
        if scores is not None:
            self._scores = {
                column: float(score) if score is not None else float('nan')
                for column, score in zip(self.columns, scores)
            }

        if self.columns_to_drop:
            for column in self.columns_to_drop:
                score = self._scores.get(column)
                if score is None or pd.isna(score):
                    self.explanations.append(
                        f"Dropped column **`{column}`** because it was not in "
                        f"the top **{k_value}** features."
                    )
                else:
                    self.explanations.append(
                        f"Dropped column **`{column}`** because it was not in "
                        f"the top **{k_value}** features (score={score:.4f})."
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

        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns or dataset.X.empty:
            return False
        if dataset.X[columns].isna().any().any():
            return False

        if self._resolve_k(len(columns)) < 1:
            return False

        score_func = self.get_config('score_func')
        if dataset.type_of_target == 'continuous':
            return score_func == 'mutual_info'

        if dataset.type_of_target not in [
            'binary', 'multiclass', 'multilabel-indicator'
        ]:
            return False

        if score_func == 'chi2':
            return not (dataset.X[columns] < 0).any().any()

        return score_func in ['f_classif', 'mutual_info', 'chi2']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
