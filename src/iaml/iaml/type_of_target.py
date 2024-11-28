"""
Try to figure out the type of target
"""
from typing import List, Tuple, Any
import numpy as np
from sklearn.utils.multiclass import type_of_target as sk_type_of_target


def type_of_target(y: List) -> str:
    """Try to figure out the type of target

    :param List y: Dataset's target

    :return: type of target (binary, continuous, multi-label, etc.)
    """
    y = np.array(y)
    
    # Survival target -> list[tuple[bool, int]]
    if is_survival(y):
        return 'survival'
    if y.dtype == float and len(np.unique(y)) > len(y)*0.2:
        return 'continuous'
    
    return sk_type_of_target(y)

def is_survival(y: List[Tuple[Any, Any]]) -> bool:
    """Checks if the input data represents survival data.
    
    Survival data is expected to be a list of tuples where:
    - The first element of each tuple is binary-like
    - The second element of each tuple is numeric (int or float).
    
    :param List[Tuple[Any, Any]] y: each tuple contains two elements (binary-like, numeric).
    
    :return: True if the data meets the survival data requirements, False otherwise.
    """
    
    # Ensure all elements are tuples of length 2
    if not all(isinstance(row, tuple) and len(row) == 2 for row in y):
        return False
    
    # Check if the first element of all tuples is binary-like
    first_elements = [row[0] for row in y]
    if any(not isinstance(element, (bool, int))
            for element in first_elements):
        return False
    
    # Check if the second element of all tuples is numeric (int or float)
    if not all(isinstance(row[1], (int, float)) for row in y):
        return False
    
    return True
