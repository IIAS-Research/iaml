from ..dataset import Dataset
from enum import Enum

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    NEVER = 4

class Step:
    def __init__(self):
        None
        
    @classmethod
    def priorize(cls, data: Dataset) -> Priority:
        return Priority.NEVER
        
    @classmethod
    def prepare(cls, data: Dataset) -> list:
        return [cls()]
    
    @classmethod
    def reusable(cls) -> bool:
        False
        
    def run(self, data: Dataset) -> Dataset:
        return data