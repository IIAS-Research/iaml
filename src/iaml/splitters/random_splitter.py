"""Allow to split randomly a dataset to train/test"""
from typing import Iterator
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from ..dataset import Dataset


def random_splitter(
    dataset: Dataset,
    ratio: float = 0.2,
    random_state: int = 42) -> Iterator[tuple['Dataset', 'Dataset']]:
    """Allow to split randomly a dataset to train/test
    
    :param Dataset dataset: The dataset to split.
    :param float, optional ratio: The train/test ratio. Default to 0.2.
    :param int, optional random_state: The random seed used. Default to 42.
    :return: Iterator of tuples of train/test Dataset objects
    """
    kwargs = {}
    if dataset.has_groups:
        kwargs['groups'] = dataset.groups
        splitter = GroupShuffleSplit(1, test_size=ratio, random_state=random_state)
    else:
        splitter = ShuffleSplit(1, test_size=ratio, random_state=random_state)

    for ds_train, ds_test in dataset.split(splitter.split, **kwargs):
        yield (ds_train, ds_test)
