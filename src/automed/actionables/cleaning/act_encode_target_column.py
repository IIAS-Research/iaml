"""
[STEP] Encode string categorical target column to numeric
"""

import numpy as np
from ...actionable import Actionable
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('cleaning')
class ActCategoryStringToNumeric(Actionable):
    """
    Encode categorical target column to numeric
    """
    name = 'Encode categorical target column to numeric'
    description = 'Encode categorical target column to numeric'
    
    def __init__(self):
        self.configuration:dict = {}
        self.columns_to_encode:list[str] = None
    
    #def fit(self, dataset:Dataset) -> Actionable:
    #    """
    #    Find column to convert
    #    
    #    Args:
    #         dataset (Dataset): Data to fit on
    #         
    #    Returns:
    #          Candidate: Transformed candidate (with updated pipeline)
    #    """
#
    #    self.column_to_encode = dataset.y
    #    if not isinstance(self.column_to_encode, pd.Series):
    #        self.column_to_encode = pd.Series(self.column_to_encode)
    #    #print(self.column_to_encode)
    #    
    #    return self
    
    # def transform(self, y:pd.Series) -> pd.Series:
    #     """
    #     Transform y to Series and convert categorical target column of candidate dataset to numeric
        
    #     Args:
    #         y (pd.Series): Target column to transform
            
    #     Returns:
    #         pd.Series: Transformed target column
    #     """
    #     print(type(y))
    #     y = pd.Series(y)
    #     if y.dtype == 'object' or y.dtype == 'bool':
    #         encoded_target = y.astype('category').cat.codes
    #         print('transform, encode traget, stringtonumeric:', encoded_target)
    #         return encoded_target
    #     return y  # If no conversion is needed, return original target column
    

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