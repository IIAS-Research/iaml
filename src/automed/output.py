from copy import copy
from dataclasses import dataclass
from typing import Iterator

from .dataset import Dataset


class Output:
    
    def __init__(self, dataset:Dataset=None, metrics:list=[], auto_pipeline=None, stack_list=[], main_metric=None):
        from .auto_pipeline import AutoPipeline # Here to avoid circular import. TODO -> Something better to do ?
        
        self.__resample_stack = []

        self.dataset = dataset
        self.metrics = copy(metrics)
        self.pipeline = auto_pipeline or AutoPipeline() # Pipeline
        
        if main_metric == None and dataset and dataset.type_of_target:
            self.main_metric = 'r2_score' if 'continuous' in dataset.type_of_target else 'balanced_accuracy'
        else:
            self.main_metric = main_metric
            
        self.computed_metrics = {}
        self.stacked_path = copy(stack_list)
        
    def add_stack(self, stack):
        self.stacked_path.append(stack)
        
    def get_main_metric_value(self):
        if self.main_metric in self.computed_metrics:
            return self.computed_metrics[self.main_metric]
        else:
            return -1
    
    def __gt__(self, other):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() > other.get_main_metric_value()
        else:
            if self.computed_metrics:
                return True
            if other.computed_metrics:
                return False
            
            return id(self) > id(other)
            
    def __lt__(self, other):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() < other.get_main_metric_value()
        else:
            if self.computed_metrics:
                return False
            if other.computed_metrics:
                return True
            
            return id(self) < id(other)
            
    def __eq__(self, other):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() == other.get_main_metric_value()
        else:
            return id(self) == id(other)
        
    def to_output(self, dataset:Dataset=None, metrics=None, auto_pipeline=None):
        return Output(
            (dataset or self.dataset or Dataset()),
            metrics or copy(self.metrics),
            auto_pipeline or self.pipeline.copy(),
            stack_list=self.stacked_path)
        
    def to_input(self, dataset:Dataset=None, metrics=None, auto_pipeline=None):
        return Input(
            (dataset or self.dataset or Dataset()),
            metrics or copy(self.metrics),
            auto_pipeline or self.pipeline.copy(),
            stack_list=self.stacked_path)
    
    def add_transform(self, instance) -> 'Output':
        """
        Transforms the dataset using the provided function, and adds it to the
        stack of functions to be applied before prediction.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        self.dataset.transform(instance.transform)
            
        if self.pipeline is not None and instance.transform is not None and callable(instance.transform):
            self.pipeline.add_to_stack(instance)
        
        return self.to_output()
    
    
    def add_resample(self, instance) -> 'Output':
        """
        Resample the dataset using the resample method of provided instance.
        The method will be applied to the dataset just before training,
        after splitting into train and test.
        """
        self.dataset.resample(instance.resample)
        
        return self.to_output()
    
    def set_model(self, instance) -> 'Output':
        """
        Sets the resulting model of the pipeline to this output.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        return self.to_output(auto_pipeline=self.pipeline.set_model(instance))
    
    def add_metric(self, metric) -> None:
        self.metrics.append(metric)
        
    def __str__(self):
        str_out = ""
        if self.metrics:
            str_out = str_out + str(self.metrics) + " "
        if self.pipeline:
            str_out = str_out + str(self.pipeline) + " "
            
        return str_out
    
    def evaluate(self, dataset:Dataset, force:bool=False):
        if not(self.pipeline.have_model):
            return None
        
        training_stage = dataset.splitted # If dataset is splitted -> We here in the training stage
        
        if training_stage and (not(force) and self.computed_metrics):
                return self.computed_metrics
            
        current_compute = {}
        for metric in self.metrics:
            result = dataset.compute_metric(self.pipeline, metric)
            current_compute[metric.__str__()] = result
            
        if training_stage:
            self.computed_metrics = current_compute
            
        return current_compute
    
    def log(self, test):
        pass
    
    def save(self):
        pass
    
    def explain(self):
        return list(map(lambda stack: stack.explain(), self.stacked_path))
    
    
# Alias for Output
class Input(Output):
    pass