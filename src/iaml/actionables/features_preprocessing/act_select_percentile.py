"""[STEP] Decompose features with SelectPercentile"""

import textwrap
from collections.abc import Callable

import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


def _is_numeric_matrix(values: pd.DataFrame) -> bool:
    if values.empty:
        return False
    for column in values.columns:
        if not pd.api.types.is_numeric_dtype(values[column]):
            return False
    return not values.isna().any().any()


def _resolve_score_func(value) -> Callable | None:
    if callable(value):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == 'chi2':
            return chi2
        if lowered == 'f_classif':
            return f_classif
    return None


@is_step('features_preprocessing')
class ActSelectPercentile(Actionable):
    """[STEP] Preprocess with SelectPercentile"""

    name: str = "Preprocess with SelectPercentile"
    _usage: str = "Use when you need fast univariate feature selection by percentile, rather than ActKernelPCA. Applicable to non-negative features with classification targets (chi2 or f_classif). Avoid when you want feature engineering via clustering like ActKMeansFeatures."
    _description: str  = textwrap.dedent('''\
        SelectPercentile is a tool that helps choose important features from a
        group of variables by looking at how well each one predicts the outcome.''')
    _description_long: str  = textwrap.dedent('''\
        SelectPercentile is a feature selection technique used in machine
        learning. It works by assigning scores to each feature based on how well it predicts
        the outcome. Then, it selects only the top-scoring percentage of features.
        This helps reduce the number of variables while keeping the most informative ones.''')

    def __init__(self):
        self.configuration = {
            'score_func': {
                'description': 'function taking two arrays X and y, \
                    and returning a pair of arrays',
                'default': chi2,
                'categorical': [chi2, f_classif]
                },
            'percentile': {
                'description': 'Percent of features to keep.',
                'default': 50.0,
                'range': [1.0, 99.0]
                }
            }

        self.optimizable: bool = True
        self.preprocessor: SelectPercentile | None = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = None
        if dataset.y is None or not _is_numeric_matrix(dataset.X):
            return self

        score_func = _resolve_score_func(self.get_config('score_func'))
        if score_func is None:
            return self

        params = self.passthrough_parameters()
        params['score_func'] = score_func
        self.preprocessor = SelectPercentile(**params)
        self.preprocessor.fit(dataset.X, dataset.y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply SelectPercentile

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None:
            return X
        return pd.DataFrame(self.preprocessor.transform(X))

    def suitable(self, dataset: Dataset) -> bool:
        # Negative values are not supported
        if dataset.y is None or dataset.type_of_target is None:
            return False
        if not _is_numeric_matrix(dataset.X):
            return False
        score_func = _resolve_score_func(self.get_config('score_func'))
        if score_func is None:
            return False
        if dataset.type_of_target not in [
            'binary', 'multiclass', 'multilabel-indicator'
        ]:
            return False
        if score_func == chi2:
            return not (dataset.X < 0).any().any()
        return True

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
