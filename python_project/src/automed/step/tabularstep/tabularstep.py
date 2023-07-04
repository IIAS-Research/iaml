from ..step import Step, Priority
from ...dataset import Dataset

class TabularStep(Step):
    def __init__(self):
        None
        
    @classmethod
    def priorize(self, dataset: Dataset) -> Priority:
        return Priority.NEVER
        
    @classmethod
    def prepare(cls, dataset: Dataset) -> list:
        return [cls()]
        
    def run(self, dataset: Dataset) -> Dataset:
        return dataset