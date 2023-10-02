# from pipeline import Pipeline
from . import MetaStep
from . import Cleaner

class TabularClassifier(MetaStep): #MetaStep
    from stepmethods import *
    
    def __init__(self):
        super().__init__()
        print("Init TabularClassifier")
        print(super())
        
    @classmethod
    def __include_step(cls) -> list:
        return [Cleaner]
        # return [Merger, Cleaner, Labeler, Spliter, Classifier, Evaluator]
        
    @classmethod
    def forced_priority(cls) -> list:
        return []
        
    @classmethod
    def __exclude_step(cls) -> list:
        return []
    