from ..step import Step, Priority
from ...dataset import Dataset
from ...pipeline import Pipeline

class MetaStep(Step):
    def __init__(self):
        self.generate_pipeline()
        self.output = None
        
    @classmethod
    def available_steps(cls) -> list:
        return [step for step in cls.__include_step() if step not in cls.__exclude_step()]
        
    
    @classmethod
    def __include_step(cls) -> list:
        return []
        
    @classmethod
    def __exclude_step(cls) -> list:
        return []
    
    @classmethod
    def available_specialization(cls):
        return cls.__subclasses__()
    
    def specialize(self, type):
        None
        
    def generate_pipeline(self):
        self.pipeline = Pipeline(self.available_steps())
        
        
    @classmethod
    def prepare(cls, dataset: Dataset) -> list:
        return [cls()]
        
    def run(self, dataset: Dataset) -> Dataset:
        self.pipeline.run(dataset)
        return self.pipeline.outputs()