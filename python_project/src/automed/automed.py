from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .output import Output
from .dataset import Dataset
from .metric import Metric

from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep

# Default Actionables
from .actionables import *

# Default Wrappers
from .wrapper import *

class AutoMed:
    output:list = None
    input:Dataset = None
    
    def __init__(self, dataset:Dataset=None):
        self.output = None
        self.input = Output(dataset, None, None)
        self.first_step = None
    
    def load_pipe(self, pipe):
        pass
    
    def autosklearn_load(self, time=30):
        self.first_step = MetaOrderedStep()   
        self.first_step.add_step(RandomSplit()) 
        
        sklearn = ActAutoSKLearn()
        sklearn.configure_one(0, 'running_time', time)
        
        self.first_step.add_step(sklearn) 
    
    def debug_load(self, only=None):
        if only:
            self.first_step = MetaOrderedStep()
            self.first_step.add_step(RandomSplit())
            self.first_step.add_step(MetaStep(tag=only))
            self.first_step.add_step(MetaStep(tag='features_selection'))
            
        else: 
            self.first_step = MetaOrderedStep()
            self.first_step.add_step(RandomSplit())
            self.first_step.add_step(MetaStep(tag='cleaning'))
            self.first_step.add_step(MetaStep(tag='features_selection'))
            self.first_step.add_step(MetaStep(tag='normalize'))
            self.first_step.add_step(MetaExplorerStep(tag='learning', wrap=WrapIterativeGridSearch))
            # self.first_step.add_step(MetaExplorerStep(tag='boosting'))
            
    @property
    def dataset(self):
        return self.input.dataset
    @property
    def metric(self):
        return self.input.metric
    @property
    def model(self):
        return self.input.model
    
    def run(self, callback=None):
        copied_input = self.input.to_output() # Avoid input to be edited by steps
        self.output = self.first_step.run(copied_input, callback=callback)
        
        # Order ouputs 
        self.output.sort(key=lambda output: output.metric.compute(output), reverse=True)
        
        
        return self.output
    
    