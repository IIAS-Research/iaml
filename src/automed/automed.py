from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .output import Output, Input
from .dataset import Dataset
from .metric import Metric
from .model import Model
from .destroyer import Destroyer
from .worker_manager import WorkerManager

from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep

# Default Actionables
from .actionables import *

# Default Wrappers
from .wrapper import *

# cuDF pandas acceleration
try:
    import cudf.pandas
    cudf.pandas.install()
    Logger().log('cuDF is installed: using cuDF pandas accelerator mode.')
except Exception as e:
    Logger().log('cuDF not found: falling back to standalone pandas.')

# Main class of the package
# Usefull to create & run pipeline
class AutoMed:
    output:list = None # Outputs of the pipeline after run
    input:Input = None # Input Data
    
    def __init__(self, dataset:Dataset=None, max_workers:int=None, quiet=False):
        self.output = None
        self.input = Input(dataset) # Gerenate Input object from Dataset
        self.first_step = None # Will be the first Step of the pipeline (probably a MetaStep)
        
        Logger().set_quiet(quiet)

        WorkerManager(max_workers=max_workers)
    
    
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
        # step.add_step(MetaStep(tag='features_selection'))
        step.add_step(MetaStep(tag='normalize'))
        step.add_step(MetaStep(tag='metric'))
        step.add_step(ActTPLOT())

        return step
    
    # DEBUG -> Testing purpose.
    def debug_pipeline(self, only=None, use_destroyer=False, fast=False):
        step = MetaOrderedStep()
        step.add_step(RandomSplit())

        if only:
            step.add_step(MetaStep(tag=only))
        else: 
            step.add_step(MetaStep(tag='cleaning'))
            step.add_step(MetaStep(tag='features_selection'))
            step.add_step(MetaStep(tag='normalize'))
            step.add_step(MetaStep(tag='metric'))
            
            learning_tag = 'fast_learning' if fast else 'learning'
            if use_destroyer:
                step.add_step(MetaExplorerStep(tag=learning_tag, wrap=WrapGeneticGridSearch, destroyer=Destroyer()))
            else:
                step.add_step(MetaExplorerStep(tag=learning_tag, wrap=WrapGeneticGridSearch))
                
        
        return step

    def autosklearn_load(self, time=None):
        self.first_step = self.autosklearn_pipeline(time)

    def tplot_load(self):
        self.first_step = self.tplot_pipeline()

    def debug_load(self, only=None, use_destroyer=False, fast=False):
        self.first_step = self.debug_pipeline(only, use_destroyer, fast)
    
    
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
        self.input.dataset.debug_force_monolabel() # TODO REMOVE Quand le monolabel sera obligatoire 
        
        copied_input = self.input.to_input() # Avoid input to be edited by futures steps
        with Logger().progress as progress:
            step_count = self.first_step.count_steps()
            task = progress.add_task('running steps...', total=step_count)
            def progress_callback(step: Step):
                progress.update(task, advance=1)
                if callback is not None:
                    return callback(step)
            # RUN!
            self.output = self.first_step.run(copied_input, callback=progress_callback)
            progress.update(task, completed=step_count)
            
        
        # Order ouputs according the first metric
        self.output.sort(reverse=True)
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