from ...step import Priority
from ...dataset import Dataset
from ..metastep import MetaStep
from ..tabularstep.dropnastep import DropNaStep

# TODO : Rename TabularCleaner
# TODO : Find a way to auto discover new step
class Cleaner(MetaStep):
    def __init__(self) -> None:
        self.source_data = None
        self.generate_pipeline()
    
    @classmethod
    def priorize(cls, dataset: Dataset) -> Priority:
        return Priority.HIGH
        
    # def run(self, dataset: Dataset) -> Dataset:
    #     return self.pipeline.run(dataset)
    
    @classmethod
    def available_steps(self) -> list:
        return [step for step in (super().available_steps() + self.included_step()) if step not in self.excluded_step()]
        
    @classmethod
    def included_step(self) -> list:
        return [DropNaStep]   
    
    @classmethod
    def excluded_step(self) -> list:
        return []
        
    
    
    
