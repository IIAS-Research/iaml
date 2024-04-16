import numpy as np
from sklearn.utils.multiclass import type_of_target as sk_type_of_target

def type_of_target(y) -> str:
    y = np.array(y)
    if y.dtype == float and len(np.unique(y)) > len(y)*0.2:
        return 'continuous'
    return sk_type_of_target(y)
