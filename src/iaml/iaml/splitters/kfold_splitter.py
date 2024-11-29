"""Allow to split a dataset into n folds to compute crossvalidation"""
from typing import Iterator
from sklearn.model_selection import KFold as SKKFold, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from ..dataset import Dataset


def kfold_splitter(dataset: Dataset, nb_folds: int = 5) -> Iterator[tuple['Dataset', 'Dataset']]:
    """Allow to split a dataset into n folds to compute crossvalidation

    :param Dataset dataset: The dataset to split.
    :param int, optional nb_folds: The number of folds to create. Default to 5.
    :return: Iterator of tuples of train/test Dataset objects
    """
    kwargs = {}

    if dataset.type_of_target in ['binary', 'multiclass']:
        if dataset.has_groups:
            kfold = StratifiedGroupKFold(nb_folds)
            kwargs['groups'] = dataset.groups
        else:
            kfold = StratifiedKFold(nb_folds)
    else:
        if dataset.has_groups:
            kfold = GroupKFold(nb_folds)
            kwargs['groups'] = dataset.groups
        else:
            kfold = SKKFold(nb_folds)

    for ds_train, ds_test in dataset.split(kfold.split, **kwargs):
        yield (ds_train, ds_test)
