from .dataset import Dataset
from copy import copy

class Output:
    
    def __init__(self, dataset:Dataset=None, metrics:list=[], model=None, stack_list=[], main_metric=None):
        from .model import Model # Here to avoid circular import. TODO -> Something better to do ?
        
        self.dataset = dataset
        self.metrics = copy(metrics)
        self.model = model or Model() # Pipeline
        
        if main_metric == None and dataset and dataset.type_of_target:
            self.main_metric = 'r2_score' if 'continuous' in dataset.type_of_target else 'balanced_accuracy'
        else:
            self.main_metric = main_metric
            
        self.computed_metrics = {}
        self.stacked_path = copy(stack_list)
        
    def add_stack(self, stack):
        self.stacked_path.append(stack)
        
    def get_main_metric_value(self):
        if self.main_metric in self.evaluate():
            return self.computed_metrics[self.main_metric]
        else:
            return -1
    
    def __gt__(self, other):
        if self.evaluate() and other.evaluate():
            return self.get_main_metric_value() > other.get_main_metric_value()
        else:
            if self.evaluate():
                return True
            if other.evaluate():
                return False
            
            return id(self) > id(other)
            
    def __lt__(self, other):
        if self.evaluate() and other.evaluate():
            return self.get_main_metric_value() < other.get_main_metric_value()
        else:
            if self.evaluate():
                return False
            if other.evaluate():
                return True
            
            return id(self) < id(other)
            
    def __eq__(self, other):
        if self.evaluate() and other.evaluate():
            return self.get_main_metric_value() == other.get_main_metric_value()
        else:
            return id(self) == id(other)
        
    def to_output(self, dataset:Dataset=None, metrics=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()),
            metrics or copy(self.metrics),
            model or self.model.copy(),
            stack_list=self.stacked_path)
        
    def to_input(self, dataset:Dataset=None, metrics=None, model=None):
        return Input(
            (dataset or self.dataset or Dataset()),
            metrics or copy(self.metrics),
            model or self.model.copy(),
            stack_list=self.stacked_path)

    def transform_dataset(self, instance, only_train=False):
        """
        Transforms the dataset using the provided function, and adds it to the
        stack of functions to be applied before prediction.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        # TODO -> Check if still pickable after refacto
            
        self.dataset.apply(instance.transform, only_train)
        if self.model is not None and instance.transform is not None and callable(instance.transform):
            self.model.add_to_stack(instance)
        
        return self.to_output()
    
    def set_model(self, instance):
        """
        Sets the resulting model of the pipeline to this output.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        return self.to_output(model=self.model.set_model(instance))
    
    def add_metric(self, metric) -> None:
        self.metrics.append(metric)
        
    def __str__(self):
        str_out = ""
        if self.metrics:
            str_out = str_out + str(self.metrics) + " "
        if self.model:
            str_out = str_out + str(self.model) + " "
            
        return str_out
    
    def evaluate(self, force=False):
        if not(self.model.have_model):
            return None
        
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