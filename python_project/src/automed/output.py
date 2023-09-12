from dataclasses import dataclass
from .dataset import Dataset

@dataclass
class Output:
    # dataset:Dataset = None
    # metric = None
    # model = None
    # stacked_log = []
    
    def __init__(self, dataset:Dataset, metric, model):
        self.dataset = dataset
        self.metric = metric
        self.model = model
        
    def to_output(self, dataset:Dataset=None, metric=None, model=None):
        return Output(
            (dataset or self.dataset or Dataset()).copy(),
            metric or self.metric,
            model or self.model)
        
    def log(self, test):
        pass
    
    def save(self):
        pass