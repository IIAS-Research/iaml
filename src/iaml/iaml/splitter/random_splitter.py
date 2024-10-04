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
    kwargs = {}
    if dataset.has_groups:
        kwargs['groups'] = dataset.groups
        splitter = GroupShuffleSplit(1, test_size=ratio, random_state=random_state)
    else:
        splitter = ShuffleSplit(1, test_size=ratio, random_state=random_state)

    for ds_train, ds_test in dataset.split(splitter.split, **kwargs):
        yield (ds_train, ds_test)
