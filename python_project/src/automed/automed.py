from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .output import Output, Input
from .dataset import Dataset
from .metric import Metric

from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep

# Default Actionables
from .actionables import *

# Default Wrappers
from .wrapper import *

# Main class of the package
# Usefull to create & run pipeline
class AutoMed:
    output:list = None # Outputs of the pipeline after run
    input:Input = None # Input Data
    
    def __init__(self, dataset:Dataset=None):
        self.output = None
        self.input = Input(dataset, None, None) # Gerenate Input object from Dataset
        self.first_step = None # Will be the first Step of the pipeline (probably a MetaStep)
    
    # Load any king of pipe
    def load_pipe(self, pipe):
        pass # TODO
    
    # DEBUG -> Testing purpose
    def autosklearn_load(self, time=30):
        self.first_step = MetaOrderedStep()   
        self.first_step.add_step(RandomSplit()) 
        
        sklearn = ActAutoSKLearn()
        sklearn.configure_one(0, 'running_time', time)
        
        self.first_step.add_step(sklearn) 
    
    # DEBUG -> Testing purpose. To replace when load_pipe is working
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
    
    
    ##################
    ### PROPERTIES ###
    ##################
    
    # Dataset from input data
    @property
    def dataset(self):
        return self.input.dataset
    
    # Metric from input data
    @property
    def metric(self):
        return self.input.metric
    
    # Model from input data
    @property
    def model(self):
        return self.input.model
    
    ###########
    ### RUN ###
    ###########
    
    # Execute all the pipeline steps
        # Callback -> Will be call after each step 
    def run(self, callback=None):
        copied_input = self.input.to_input() # Avoid input to be edited by futures steps
        
        # RUN !
        self.output = self.first_step.run(copied_input, callback=callback)
        
        # Order ouputs according results
        self.output.sort(key=lambda output: output.metric.compute(output), reverse=True)
        
        return self.output
    
    ########################
    #### CONFIGURATIONS ####
    ########################
    
    # Return a dict with configurations of all steps. 
    def all_configurations(self):
        pass # TODO
    
    # Configure one to many steps with a dict configurations 
    def configure_all(self, configs):
        pass # TODO
    
    