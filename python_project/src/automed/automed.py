from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .output import Output, Input
from .dataset import Dataset
from .metric import Metric
from .destroyer import Destroyer

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
    
    
    # Load any kind of pipeline
    def load_pipeline(self, pipeline: dict) -> None:
        self.first_step = Step.from_pipeline(pipeline)
    
    # DEBUG -> Testing purpose
    def autosklearn_pipeline(self, time=30):
        step = MetaOrderedStep()   
        step.add_step(RandomSplit()) 
        
        sklearn = ActAutoSKLearn()
        sklearn.configure_one(0, 'running_time', time)
        
        step.add_step(sklearn)

        return step 
        
    # DEBUG -> Testing purpose
    def tplot_pipeline(self):
        step = MetaOrderedStep()
        step.add_step(RandomSplit())
        step.add_step(MetaStep(tag='cleaning'))
        step.add_step(MetaStep(tag='features_selection'))
        step.add_step(MetaStep(tag='normalize'))
        step.add_step(ActTPLOT())

        return step
    
    # DEBUG -> Testing purpose.
    def debug_pipeline(self, only=None, use_destroyer=False):
        step = MetaOrderedStep()
        step.add_step(RandomSplit())

        if only:
            step.add_step(MetaStep(tag=only))
            step.add_step(MetaStep(tag='features_selection'))
            
        else: 
            step.add_step(MetaStep(tag='cleaning'))
            step.add_step(MetaStep(tag='features_selection'))
            step.add_step(MetaStep(tag='normalize'))
            if use_destroyer:
                step.add_step(MetaExplorerStep(tag='learning', wrap=WrapGeneticGridSearch, destroyer=Destroyer()))
            else:
                step.add_step(MetaExplorerStep(tag='learning', wrap=WrapGeneticGridSearch))
                
            # self.first_step.add_step(MetaExplorerStep(tag='boosting'))
        
        return step

    def autosklearn_load(self, time=None):
        self.first_step = self.autosklearn_pipeline(time)

    def tplot_load(self):
        self.first_step = self.tplot_pipeline()

    def debug_load(self, only=None, use_destroyer=False):
        self.first_step = self.debug_pipeline(only, use_destroyer)
    
    
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
    
    def json_pipeline(self):
        return self.first_step.json_pipeline()
    
    # Return a dict with configurations of all steps. 
    def all_configurations(self):
        return self.first_step.all_configurations()
    
    # Configure one to many steps with a dict configurations 
    def configure_all(self, configs):
        all_steps = self.__all_steps()
        
        for step_id, config in configs.items():
            current_step = self.__find_step_by_id(all_steps, step_id)
            if current_step:
                for key, value in config:
                    current_step.configure_one(0, key, value)
    
    def __all_steps(self):
        return self.first_step + self.first_step.all_steps()
    
    def __find_step_by_id(self, step_list, step_id):
        for step in step_list:
            if id(step) == step_id:
                return step
        return False
        
    
    
    
    