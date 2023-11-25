from dataclasses import dataclass
from .dataset import Dataset
from copy import copy

@dataclass
class Output:
    # dataset:Dataset = None
    # metric = None
    # model = None
    # stacked_log = []
    
    def __init__(self, dataset:Dataset=None, metric=None, model=None, stack_list=[]):
        self.dataset = dataset
        self.metric = metric
        self.model = model
        self.computed = None
        self.stacked_path = copy(stack_list)
        
    def add_stack(self, stack):
        self.stacked_path.append(stack)
        
        
    def __gt__(self, other):
        if self.computed and other.computed:
            return self.computed > other.computed
        else:
            if self.computed:
                return True
            if other.computed:
                return False
            
            return id(self) > id(other)
            
    def __lt__(self, other):
        if self.computed and other.computed:
            return self.computed < other.computed
        else:
            if self.computed:
                return False
            if other.computed:
                return True
            
            return id(self) < id(other)
            
    def __eq__(self, other):
        if self.computed and other.computed:
            self.computed == other.computed
        else:
            id(self) == id(other)
        
    def to_output(self, dataset:Dataset=None, metric=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()),
            metric or self.metric,
            model or self.model,
            stack_list=self.stacked_path)
        
    def to_input(self, dataset:Dataset=None, metric=None, model=None):
        return Input(
            (dataset or self.dataset or Dataset()),
            metric or self.metric,
            model or self.model,
            stack_list=self.stacked_path)
        
    def __str__(self):
        str_out = ""
        if self.metric:
            str_out = str_out + str(self.metric) + " "
        if self.model:
            str_out = str_out + str(self.model) + " "
            
        return str_out
    
    def compute(self, force=False):
        if force or not(self.computed):
            self.computed = self.metric.compute(self)
        return self.computed
        
    def log(self, test):
        pass
    
    def save(self):
        pass
    
    def explain(self):
        return list(map(lambda stack: stack.explain(), self.stacked_path))
    
    
# Alias for Output
class Input(Output):
    pass