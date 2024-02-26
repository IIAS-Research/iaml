from dataclasses import dataclass
from .dataset import Dataset
from copy import copy

@dataclass
class Output:
    # dataset:Dataset = None
    # metric = None
    # model = None
    # stacked_log = []
    
    # TODO -> main_metric -> Remove hardcoded one and add selection mechanisms 
    def __init__(self, dataset:Dataset=None, metrics=[], main_metric='balanced_accuracy', model=None, stack_list=[]):
        from .model import Model # Here to avoid circular import. TODO -> Something better to do ?
        
        self.dataset = dataset
        self.metrics = metrics
        self.main_metric = main_metric
        self.model = model or Model()
        self.computed_metrics = {}
        self.stacked_path = copy(stack_list)
        
    def add_stack(self, stack):
        self.stacked_path.append(stack)
        
    @property
    def main_metric_value(self):
        self.computed_metrics[self.main_metric]
    
    def __gt__(self, other):
        if self.computed_metrics and other.computed:
            return self.main_metric_value > other.main_metric_value
        else:
            if self.computed_metrics:
                return True
            if other.computed_metrics:
                return False
            
            return id(self) > id(other)
            
    def __lt__(self, other):
        if self.computed_metrics and other.computed_metrics:
            return self.main_metric_value < other.main_metric_value
        else:
            if self.computed_metrics:
                return False
            if other.computed_metrics:
                return True
            
            return id(self) < id(other)
            
    def __eq__(self, other):
        if self.computed_metrics and other.computed_metrics:
            self.main_metric_value == other.main_metric_value
        else:
            id(self) == id(other)
        
    def to_output(self, dataset:Dataset=None, metrics=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()),
            metrics or self.metrics,
            model or self.model.copy(),
            stack_list=self.stacked_path)
        
    def to_input(self, dataset:Dataset=None, metrics=None, model=None):
        return Input(
            (dataset or self.dataset or Dataset()),
            metrics or self.metrics,
            model or self.model.copy(),
            stack_list=self.stacked_path)

    def transform_dataset(self, function: callable = None, *args, **kw):
        """
        Transforms the dataset using the provided function, and adds it to the
        stack of functions to be applied before prediction.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        if 'only_train' in kw:
            only_train = kw['only_train']
            del kw['only_train']
        else:
            only_train = False
            
        self.dataset.apply(function, only_train, *args, **kw)
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
        return self.metrics.append(metric)
        
    def __str__(self):
        str_out = ""
        if self.metrics:
            str_out = str_out + str(self.metrics) + " "
        if self.model:
            str_out = str_out + str(self.model) + " "
            
        return str_out
    
    def evaluate(self, force=False):
        if not(self.model.have_model):
            return -1
        
        if force or not(self.computed_metrics):
            for metric in self.metrics:
                result = self.dataset.compute_metric(self.model, metric)
                self.computed_metrics[metric.__str__()] = result
                
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