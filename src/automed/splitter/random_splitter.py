"""
Allow to split randomly a dataset to train/test 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from ..dataset import Dataset

def random_splitter(dataset:Dataset, ratio=0.2, random_state=42):
    """
    Allow to split randomly a dataset to train/test 
    """
    def split(X:pd.DataFrame, y:np.array): # pylint: disable=unused-argument
        return ShuffleSplit(1, test_size=ratio, random_state=random_state).split(X)
    
    for ds_train, ds_test in dataset.split(split):
        yield (ds_train, ds_test)
