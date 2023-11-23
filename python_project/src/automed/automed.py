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
    
    def parse_pipeline_step(self, pipeline_step: dict) -> Step:
        if 'value' not in pipeline_step:
            return None

        step_conf = pipeline_step['value']

        if 'step' not in step_conf:
            raise TypeError(f'invalid pipeline: missing step attribute')

        # filter registered steps and see if given step exists
        steps = list(filter(lambda step: step.__name__ == step_conf['step'], Step.available_steps.keys()))
        if len(steps) == 0:
            raise TypeError(f'invalid step ({step_conf["step"]})')

        step = None
        if MetaStep in steps[0].__mro__:
            # step is a MetaStep
            tag = None
            if 'tag' in step_conf:
                tag = step_conf['tag']

            wrap = None
            if 'wrap' in step_conf:
                # filter step wrappers, then filter registered steps and see if given step exists
                wrap_steps = list(filter(lambda step: StepWrapper in step.__mro__ and step.__name__ == step_conf['wrap'], Step.available_steps.keys()))

                if len(wrap_steps) == 0:
                    raise TypeError(f'invalid wrap step ({step_conf["wrap"]})')

                if len(wrap_steps) > 0:
                    wrap = wrap_steps[0]

            destroyer = None
            if 'use_destroyer' in step_conf and step_conf['use_destroyer']:
                destroyer = Destroyer()

            step = steps[0](tag=tag, wrap=wrap, destroyer=destroyer)
        elif StepWrapper in steps[0].__mro__:
            # step is a StepWrapper, so its child is the wrapped step
            if 'children' not in pipeline_step or len(pipeline_step['children']) == 0:
                raise TypeError(f'invalid pipeline: missing child for step wrapper ({step_conf["step"]})')

            child = self.parse_pipeline_step(pipeline_step['children'][0])
            step = steps[0](child)
        else:
            # step is not a MetaStep
            step = steps[0]()
        
        # load each specified configuration value for each of the given steps
        if 'configuration' in step_conf:
            conf = step_conf['configuration']
            for name, c in conf.items():
                step.configure_one(0, name, c['value'])

        return step
    
    # Load any kind of pipeline
    def load_pipeline(self, pipeline: dict, first_step: bool = True) -> None:
        step = self.parse_pipeline_step(pipeline)
        if step is not None:
            if first_step:
                self.first_step = step
            else:
                self.first_step.add_step(step)
        
            if 'children' in pipeline and StepWrapper not in step.__class__.__mro__:
                if MetaExplorerStep in step.__class__.__mro__:
                    # if step is MetaExplorerStep, add the children to the same step
                    for child in pipeline['children']:
                        if 'value' in child:
                            child_step = self.parse_pipeline_step(child)
                            step.add_step(child_step)
                elif len(pipeline['children']) > 0:
                    # if step is not a MetaExplorer, add the child if any
                    self.load_pipeline(pipeline['children'][0], first_step=False)
    
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
        
    
    
    
    