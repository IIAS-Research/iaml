from copy import copy
from dataclasses import dataclass
from typing import Iterator

from .dataset import Dataset, TrainingDataset


@dataclass
class Output:
    # dataset:Dataset = None
    # metric = None
    # model = None
    # stacked_log = []
    
    def __init__(self, dataset:Dataset=None, metric=None, model=None, stack_list=[]):
        from .model import Model # Here to avoid circular import. TODO -> Something better to do ?
        
        self.__train_dataset_transform_stack = []

        self.dataset = dataset
        self.metric = metric
        self.model = model or Model()
        self.computed_metrics = None
        self.stacked_path = copy(stack_list)
        
    def add_stack(self, stack):
        self.stacked_path.append(stack)
        
        
    def __gt__(self, other):
        if self.computed_metrics and other.computed:
            return self.computed_metrics > other.computed_metrics
        else:
            if self.computed_metrics:
                return True
            if other.computed_metrics:
                return False
            
            return id(self) > id(other)
            
    def __lt__(self, other):
        if self.computed_metrics and other.computed_metrics:
            return self.computed_metrics < other.computed_metrics
        else:
            if self.computed_metrics:
                return False
            if other.computed_metrics:
                return True
            
            return id(self) < id(other)
            
    def __eq__(self, other):
        if self.computed_metrics and other.computed_metrics:
            self.computed_metrics == other.computed_metrics
        else:
            id(self) == id(other)
        
    def to_output(self, dataset:Dataset=None, metric=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()),
            metric or self.metric,
            model or self.model.copy(),
            stack_list=self.stacked_path)
        
    def to_input(self, dataset:Dataset=None, metric=None, model=None):
        return Input(
            (dataset or self.dataset or Dataset()),
            metric or self.metric,
            model or self.model.copy(),
            stack_list=self.stacked_path)

    def to_training_inputs(self, splitter: callable, *args, **kw) -> Iterator['TrainingInput']: # cast to TrainingInput
        for indexes in splitter(*args, **kw):
            if len(indexes) == 2:
                i_train, i_test = indexes

                X_train = self.dataset.X.iloc[i_train].copy()
                X_test = self.dataset.X.iloc[i_test].copy()
                Y_train = self.dataset.Y.iloc[i_train].copy()
                Y_test = self.dataset.Y.iloc[i_test].copy()
            else:
                iX_train, iX_test, iY_train, iY_test = indexes

                X_train = self.dataset.X.iloc[iX_train].copy()
                X_test = self.dataset.X.iloc[iX_test].copy()
                Y_train = self.dataset.Y.iloc[iY_train].copy()
                Y_test = self.dataset.Y.iloc[iY_test].copy()

            for (transformation, args, kw) in self.__train_dataset_transform_stack:
                X_train, Y_train = transformation(*args, **kw)

            yield self.to_input(TrainingDataset(X_train, X_test, Y_train, Y_test))

    def transform_dataset(self, function: callable = None, *args, only_train: bool = False, **kw):
        """
        Transforms the dataset using the provided function, and adds it to the
        stack of functions to be applied before prediction.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.

        If `only_train` is set to `True`, the function will be applied to the
        dataset just before training, after splitting into train and test.
        """
        if only_train:
            self.__train_dataset_transform_stack.append((function, args, kw))
        else:
            self.dataset.apply(function, *args, **kw)
            
        if self.model is not None and function is not None and callable(function):
            self.model.add_to_stack(function, *args, **kw)
        
        return self.to_output()
    
    def set_model(self, model, function: callable = None, *args, **kw):
        """
        Sets the resulting model of the pipeline to this output.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        return self.to_output(model=self.model.set_model(model, function, *args, **kw))
    
    def add_metric(self, metric):
        return self.to_output(metric=metric)
        
    def __str__(self):
        str_out = ""
        if self.metric:
            str_out = str_out + str(self.metric) + " "
        if self.model:
            str_out = str_out + str(self.model) + " "
            
        return str_out
    
    def evaluate(self, force=False):
        if not(self.model.have_model):
            return -1
        
        if force or not(self.computed_metrics):
            self.computed_metrics = self.dataset.compute_metric(self.model, self.metric)
            
        return self.computed_metrics
        
    def log(self, test):
        pass
    
    def save(self):
        pass
    
    def explain(self):
        return list(map(lambda stack: stack.explain(), self.stacked_path))
    
    
# Alias for Output
class Input(Output):
    pass


class TrainingInput(Input):
    dataset: TrainingDataset
