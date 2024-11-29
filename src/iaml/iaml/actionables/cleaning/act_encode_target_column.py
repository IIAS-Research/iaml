"""
[STEP] Encode string categorical target column to numeric
"""
# Disabled for now
# TODO -> Rework.
# - We can't use 'y' in transform
# - Probably must not be a step
# - Have a robust mapping
# - IAMLPipeline must be able to reverse the mapping after prediction
#       (otherwise, outputs have no sense)

import textwrap
import numpy as np
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


# @is_step('cleaning')
@is_step('disabled')
class ActCategoryStringToNumeric(Actionable):
    """
    Encode categorical target column to numeric
    """
    name = 'Textual Category To Numeric Value'
    description ='Encode categorical text data column to numeric value'
    description_long = textwrap.dedent('''\
        Retrieve all unique values from a column, then transform those values to a numeric type.
        Exemple: If a column contain 3 uniques values like "coffee", "tea" and "water",
        then all the coffee values will be transformed to 0, tea to 1 and water to 2.''')

    def __init__(self):
        self.column_to_encode:list[str] = None

    def fit(self, dataset:Dataset) -> Actionable:
        self.column_to_encode = [dataset.y]
        self.explanations = [
            f'Encoded column **`{np.unique(dataset.y)}`**. \
            Mapping of categorical values to numerical values:\n- `{value}`: {i}'
            for i, value in enumerate(np.unique(dataset.y)) ]
        return self

    def transform(self, y: np.array) -> np.array:
        """
        Convert categorical target columns of candidate's dataset to numeric.

        :param np.array y: Target column to transform
        :return: Transformed target column
        """
        if np.issubdtype(y.dtype, object) or np.issubdtype(y.dtype, np.bool_):
            categories = np.unique(y)
            encoded_target = np.zeros_like(y, dtype=int)
            for i, category in enumerate(categories):
                encoded_target[y == category] = i
            return encoded_target

        return y

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.4
