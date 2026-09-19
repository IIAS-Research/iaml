"""
[STEP] Decompose features with PCA
"""
import textwrap
import pandas as pd
from sklearn.decomposition import PCA
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


@is_step('features_preprocessing')
class ActPCA(Actionable):
    """[STEP] Reduce dimensions with PCA"""

    name: str = "PCA"
    _description: str = "Apply PCA for dimensionality reduction over a list of columns"
    _usage: str = "Use when you need fast linear dimensionality reduction for numeric features; consider ActKernelPCA or ActFastICA for nonlinear or independent components. Applicable to scaled numeric matrices. Avoid when features are categorical or you must keep original feature meaning."
    _description_long: str = textwrap.dedent('''\
        PCA, or Principal Component Analysis, is a dimensionality reduction technique.
        It transforms the data into a set of linearly uncorrelated components, capturing
        the maximum variance in the data with each successive component.
        This method is unsupervised, meaning it does not require labeled data,
        and is particularly useful for simplifying datasets while retaining
        as much of the underlying structure as possible.
    ''')

    def __init__(self):
        self.configuration = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 0.999,
                'range': [0.5, 0.999]
                },
            'random_state': {
                'description': 'Random State',
                'default': 42
                }
            }

        self.optimizable = True
        self.preprocessor = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = None
        if not _is_numeric_matrix(dataset.X):
            return self

        self.preprocessor = PCA(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.preprocessor is None:
            return X
        return pd.DataFrame(self.preprocessor.transform(X))

    def suitable(self, dataset: Dataset) -> bool:
        return _is_numeric_matrix(dataset.X)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
