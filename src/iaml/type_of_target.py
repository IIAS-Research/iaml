"""Try to figure out the type of target"""
from typing import Any
import numpy as np
from sklearn.utils.multiclass import type_of_target as sk_type_of_target


def type_of_target(y: list) -> str:
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

def is_survival(y: list[tuple[Any, Any]]) -> bool:
    """Checks if the input data represents survival data.
    
    Survival data is expected to be a list of tuples where:
    - The first element of each tuple is binary-like
    - The second element of each tuple is numeric (int or float).
    
    :param list[tuple[Any, Any]] y: each tuple contains two elements (binary-like, numeric).
    :return: True if the data meets the survival data requirements, False otherwise.
    """
    
    # Ensure all elements are tuples of length 2
    if not all(isinstance(row, (tuple, np.ndarray)) and len(row) == 2 for row in y):
        return False

    # Check if the first element of all tuples is binary-like
    first_elements = [row[0] for row in y]
    if not is_bool_convertible(first_elements):
        return False
    
    # Check if the second element of all tuples is numeric (int or float)
    if not all(isinstance(row[1], (int, float, np.integer, np.floating)) for row in y):
        return False

    return True

def is_bool_convertible(seq):
    for element in seq:
        if isinstance(element, (bool, np.bool_)):
            continue
        elif isinstance(element, (int, np.integer)):
            if element in (0, 1):
                continue
            else:
                return False
        elif isinstance(element, (float, np.floating)):
            if abs(element - 0.0) < 1e-9 or abs(element - 1.0) < 1e-9:
                continue
            else:
                return False
        else:
            return False
    return True
