from sklearn.model_selection import KFold as SKKFold, StratifiedKFold
from ..dataset import Dataset

def kfold_splitter(dataset:Dataset, nb_folds=5):
    if dataset.type_of_target in ['binary', 'multiclass']:
        kfold = StratifiedKFold(nb_folds)
    else:
        kfold = SKKFold(nb_folds)
        
    for ds_train, ds_test in dataset.split(kfold.split):
        yield (ds_train, ds_test)
