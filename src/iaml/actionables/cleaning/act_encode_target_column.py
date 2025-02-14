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
    _description ='Encode categorical text data column to numeric value'
    _description_long = textwrap.dedent('''\
        Retrieve all unique values from a column, then transform those values to a numeric type.
        Exemple: If a column contain 3 uniques values like "coffee", "tea" and "water",
        then all the coffee values will be transformed to 0, tea to 1 and water to 2.''')
    
    def __init__(self):
        self.configuration:dict = {}
        self.column_to_encode:list[str] = None     
        
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find column to encode and explain the transformation

        Args:
            dataset (Dataset): Data to fit on

        Returns:
            Candidate: Transformed candidate (with updated pipeline)
        """
        self.column_to_encode = [dataset.y]
        self.explanations = [f'Encoded column **`{np.unique(dataset.y)}`**. \
                            Mapping of categorical values to numerical values:\n'
                            f'- `{value}`: {i}' for i, value in enumerate(np.unique(dataset.y))]
        return self

    def transform(self, y: np.array) -> np.array:
        """
        Convert categorical target column of candidate dataset to numeric

        Args:
            y (np.ndarray): Target column to transform

        Returns:
            np.ndarray: Transformed target column
        """
        if np.issubdtype(y.dtype, object) or np.issubdtype(y.dtype, np.bool_):
            categories = np.unique(y)
            encoded_target = np.zeros_like(y, dtype=int)
            for i, category in enumerate(categories):
                encoded_target[y == category] = i
            return encoded_target
        return y

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.4
