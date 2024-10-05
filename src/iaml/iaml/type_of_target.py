"""
Try to figure out the type of target
"""
import numpy as np
from sklearn.utils.multiclass import type_of_target as sk_type_of_target

def type_of_target(y) -> str:
    """Try to figure out the type of target

    Args:
        y (list): Dataset's target

    Returns:
        str: type of target (binary, continuous, multi-label, etc.)
    """
    y = np.array(y)
    if y.dtype == float and len(np.unique(y)) > len(y)*0.2:
        return 'continuous'
    return sk_type_of_target(y)
