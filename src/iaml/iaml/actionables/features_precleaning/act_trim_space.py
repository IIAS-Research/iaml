"""[STEP] Trim spaces on each columns"""
import textwrap
from pandas.api.types import is_object_dtype
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_precleaning')
class ActTrimSpaces(Actionable):
    """
    [STEP] Trim spaces on each columns
    """
    name = "Trim columns spaces"
    _description = textwrap.dedent('''\
        This step trim spaces on each columns.
        It helps preventing errors on the dataset when casting columns with spaces''')
    _description_long = textwrap.dedent('''\
        In datasets, columns with spaces can be an issue.
        This step remove spaces in front and back of columns values.
        This ensures that columns can be casted correctly with having spaces throwing
        an error.''')

    refs = []

    def __init__(self):
        self.configuration = {
            'left_trim': {
                'description': "Trim all columns leading spaces",
                'default': True
            },
            'right_trim': {
                'description': "Trim all columns ending spaces",
                'default': True
            },
        }

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the dropping of rows with at least 40% empty columns.
        
        :param pd.DataFrame X: The dataframe to clean.
        :return: The cleaned dataframe.
        """
        if self.get_config('left_trim'):
            X = X.rename(columns=lambda x: x.lstrip())
            for col in X.columns:
                if isinstance(X[col].dtype, str) or is_object_dtype(X[col]):
                    X[col] = X[col].apply(lambda x: x.lstrip() if isinstance(x, str) else x)

        if self.get_config('right_trim'):
            X = X.rename(columns=lambda x: x.rstrip())
            for col in X.columns:
                if isinstance(X[col].dtype, str) or is_object_dtype(X[col]):
                    X[col] = X[col].apply(lambda x: x.rstrip() if isinstance(x, str) else x)
        return X

    def priorize(self, candidate: Candidate = None) -> float:
        return 1.5

    def suitable(self, dataset: Dataset) -> bool:
        cond = (
            dataset.X.apply(
                lambda x: (
                    x.astype(str).str.strip() if (isinstance(x, str) or is_object_dtype(x)) else x
                ) != x
            ).stack().any()
            or (dataset.X.columns.str.strip() != dataset.X.columns).any()
        )
        return cond
