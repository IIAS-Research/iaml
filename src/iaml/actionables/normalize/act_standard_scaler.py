"""[STEP] Standard Scaler"""
import textwrap
from sklearn.preprocessing import StandardScaler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...data_type import DataType

# TODO Fix this step
# @is_step('normalize')
class ActStandardScaler(Actionable):
    """[STEP] Standard Scaler"""

    name: str = "Standard Scaler"
    _description: str = textwrap.dedent('''\
        StandardScaler helps computers understand complex data by transforming
        it into numbers centered around zero with a standard deviation of one.''')
    _description_long: str = textwrap.dedent('''\
        StandardScaler is a machine learning technique used to standardize numerical
        features by removing the mean and scaling to unit variance.
        It works by calculating the mean and standard deviation for each feature in the training data,
        then transforming all values such that they have a mean of zero and a variance of one.
        This transformation helps to center data and is particularly useful in algorithms that assume
        normality of features, like many machine learning models.''')

    def __init__(self):
        self.columns: list[str] = None
        self.scaler: StandardScaler = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns:
            values = dataset.X[self.columns]
            self.scaler = StandardScaler()
            self.scaler.fit(values)
        else:
            self.scaler = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaler

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        if self.scaler and self.columns:
            columns = [column for column in self.columns if column in X.columns]
            if not columns:
                return X
            X[columns] = self.scaler.transform(X[columns])
        return X

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
