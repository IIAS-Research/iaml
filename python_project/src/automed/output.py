from dataclasses import dataclass
from .dataset import Dataset

@dataclass
class Output:
    # dataset:Dataset = None
    # metric = None
    # model = None
    # stacked_log = []
    
    def __init__(self, dataset:Dataset=None, metric=None, model=None):
        self.dataset = dataset
        self.metric = metric
        self.model = model
        
    def to_output(self, dataset:Dataset=None, metric=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()).copy(),
            metric or self.metric,
            model or self.model)
        
    def __str__(self):
        str_out = ""
        if self.metric:
            str_out = str_out + str(self.metric) + " "
        if self.model:
            str_out = str_out + str(self.model) + " "
            
        return str_out
        
    def log(self, test):
        pass
    
    def save(self):
        pass