# from pipeline import Pipeline
from . import MetaStep
from . import Cleaner

class TabularClassifier(MetaStep): #MetaStep
    def __init__(self):
        super().__init__()
        print("Init TabularClassifier")
        print(super())
        
    @classmethod
    def available_steps(cls) -> list:
        return [step for step in (super().available_steps() + cls.__include_step()) if step not in cls.__exclude_step()]
        
    @classmethod
    def __include_step(cls) -> list:
        return [Cleaner]
        
    @classmethod
    def __exclude_step(cls) -> list:
        return []
    