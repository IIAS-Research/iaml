"""
Allow to split randomly a dataset to train/test 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from ..dataset import Dataset

def random_splitter(dataset:Dataset, ratio: float=0.2, random_state: int=42):
    """
    Allow to split randomly a dataset to train/test 
    """
    def split(X: pd.DataFrame, y: np.array): # pylint: disable=unused-argument
        return ShuffleSplit(1, test_size=ratio, random_state=random_state).split(X)
    
    def group_split(X: pd.DataFrame, y: np.array, **kwargs): # pylint: disable=unused-argument
        return GroupShuffleSplit(
            1,
            test_size=ratio,
            random_state=random_state
        ).split(X, groups=kwargs['groups'])

    kwargs = {}
    if dataset.has_groups:
        kwargs['groups'] = dataset.groups
        return dataset.split(group_split, **kwargs)
    else:
        return dataset.split(split)
