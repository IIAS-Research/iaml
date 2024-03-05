"""
Step class is a brick used to create pipelines.
This class is not really use in Pipeline, run function doesn't do anything. 
Step class is use to create new kinds of steps by inheritance and give all
needed attributes and methods to children classes. 

There is also decorators needed to create a Step. See it under Step class.
"""

import sys
from copy import deepcopy
import pandas as pd
from .output import Output, Input

from .stack import Stack
from .logger import Logger

class Step: # pylint: disable=too-many-public-methods
    """
    Step class is use to create new kinds of steps by inheritance and give all needed
    attributes and methods to children classes. 
    
    Attributes:
        STATIC available_steps (dict) : Reference of all available Steps to create pipeline
        STATIC name (str) : Name of the step
        STATIC description (str) : Description of the step
        output (Output): Last output of the Step
        configuration (list[dict]) : Configurations of the step
        current_configuration (dict) : Current configuration enable for the next run
        self.__use_cache (bool) : Enable / Disable caching
        caches (list) : Cached result 
    """
    # Available steps. This will be filled be all the new Step loaded in Python environments
    # It will be a reference of all available Steps to create pipeline
    available_steps = {}
    
    
    # Name and description of the Step. Useful to explain pipeline to users
    name = "Step" 
    description = "Step description..."
    
    def __init__(self, *args, use_cache:bool=True, **kwargs): # pylint: disable=unused-argument
        self.output:Output = None
        self.__use_cache:bool = use_cache # Activate or not the cache of results.
        self.caches:list = [] # Cached results
        
        # Configuration of the Step. Each Step can have configurations and will save it here.
        # Step give many method to help user to configure Steps
        self.configurations:list[dict] = [{}]
        self.current_configuration:dict = {} # Current configuration
        
        # List of destroyer. Destroyers are object with a global vision of the pipeline and are able
        # to stop a pipeline branch if the results is badder then others branches
        self.destroyers:list = []
        
        self.parents_steps:list[Step] = [] # List all the previous steps before this one
        
        self.default_configurations() # Load default configuration 
        
            
    # 
    @classmethod
    def from_pipeline(cls, pipeline:dict, *args) -> 'Step':
        """
        Load any kind of Step (Step, MetaStep, Wrapper, etc) from json pipeline

        Args:
            pipeline (dict): Pipeline in JSON Format

        Raises:
            TypeError: invalid pipeline -> missing step attribute
            TypeError: invalid pipeline -> step does not exist

        Returns:
            Step: First Step of the loaded pipeline
        """
        step = None
        if 'step' not in pipeline:
            raise TypeError("invalid pipeline: missing step attribute")
        
        step_class = getattr(sys.modules['automed'], pipeline['step']) # Get class from string
        if Step in step_class.__mro__:
            if step_class == cls:
                step = cls(*args)  
            
                if 'configuration' in pipeline:
                    for name, value in pipeline['configuration'].items():
                        step.configure_one(0, name, value['value'])
            else: 
                step = step_class.from_pipeline(pipeline)
        else: 
            raise TypeError('invalid pipeline: step does not exist')
        
        return step
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None: # pylint: disable=unused-argument
        """
        WIll always raise NotImplementedError.

        Args:
            X (pd.DataFrame): X Data
            y (pd.DataFrame): Y data

        Raises:
            NotImplementedError: AutoMed Step can't be fit this way. You have to use AutoMed.run()
        """
        raise NotImplementedError("AutoMed Step can't be fit. You have to use AutoMed.run()")
        
    def __str__(self):
        return self.name
    
    def suitable(self, input_data:Input) -> bool: # pylint: disable=unused-argument
        """
        Have to be overwrote. Check if a step is suitable for a given Input

        Args:
            input_data (Input): Input to test

        Returns:
            bool: Is it suitable ?
        """
        return True
    
    def configure_child(self, child:'Step') -> 'Step':
        """
        When a Step contain others ones, this will help to setup everything
        (Transmit destroyers, increment parents steps)

        Args:
            child (Step): Step to configure

        Returns:
            Step: Configured step
        """
        if any(self.destroyers):
            child.destroyers = self.destroyers

        child.configure_parents(self)
        
        return child

    def configure_parents(self, *parents):
        """
        Backpropagates the parents to the children.
        """
        self.parents_steps.extend(map(id, parents))
    
    ################
    # Configurable #
    ################
    #
    # A actionable step can be configured by the user.
    # For example, we can configure learning rate of a machine learning step.
    #  
    # Each parameters have a name, a description and a default value.
    # Default value can be fixed or computed based on dataset
    
    def configure_one(self, config_id:int, key:str, value:any) -> None:
        """
        Configure one parameter in one configuration

        Args:
            config_id (int): Index of the configuration
            key (str): Name of the parameter
            value (any): Value to set 

        Raises:
            Exception: _description_
        """
        if key in self.configurations[config_id].keys():
            self.configurations[config_id][key]['value'] = value
        else:
            raise AttributeError(f"Configurable Key '{key}' does not exist.")
        
    
    def configure(self, config:dict, config_id:int):
        """
        Configure all parameters with a dictionary

        Args:
            config (dict): key as parameter name, value as value to set
            config_id (int, optional): Index of the configuration to update.
        """
        for key, value in config.items():
            self.configure_one(config_id, key, value)
    
    def resume_configuration(self, config_id=None) -> dict:
        """
        Resume a configuration -> No meta data, only "key: value"

        Args:
            config_id (_type_, optional): index of the configuration to resume,
                if None either the current_configuration or the first one will be choose.
                Defaults to None.

        Returns:
            dict: Configuration resume
        """
        if config_id:
            return Step.__resume_a_configuration(self.configurations[config_id])
        
        return Step.__resume_a_configuration(self.current_configuration or self.configurations[0])
    
    @classmethod
    def __resume_a_configuration(cls, config:dict):
        return {k: Step.__get_a_value(v) for k, v in config.items()}
    
    @classmethod
    def __get_a_value(cls, elem:dict) -> any:
        return elem['value'] if 'value' in elem.keys() else elem['default']
    
    def get_config(self, key:str) -> any:
        """
        Get value of a configuration key.
        Very useful to easily get configuration in inherit Step methods

        Args:
            key (str): Parameter to get

        Returns:
            any: Value of parameter
        """
        return self.resume_configuration()[key]
    
    def default_configurations(self) -> None:
        """
        Set default configuration
        Explore all configurations parameter and set default as value
        """
        for idx, current in enumerate(self.configurations):
            for key, _ in current.items():
                self.configurations[idx][key]['value'] = self.configurations[idx][key]['default']


    def keep_only_first_config(self) -> None:
        """
        Remove all configurations except the first one.
        Useful when you want a better control of step execution
        """
        self.configurations = [self.configurations[0]]
        
        
    def all_configurations(self) -> list[dict]:
        """Recursive function (last one here) to get all configurations in a pipeline

        Returns:
            list[dict]: list of all configurations
        """
        return [{
            'step_id': id(self),
            'configuration': self.configurations
        }]
        
    
    def all_step(self):
        """Recursive function (last one here) to get all steps in a pipeline

        Returns:
            list: always empty
        """
        return []
        
        
    #####################
    ## CACHING RESULTS ##
    #####################
    # Results of run() can by stored in cache to avoid compute it several time
    
    def from_cache(self, input_data:Input) -> Output:
        """
        If a previous run with same input & configuration was cached, return it
        Else return None

        Args:
            input_data (Input): Try to find this input in cache

        Returns:
            Output: Cached output or None
        """
        if not self.use_cache:
            return None
        
        for cache in self.caches:
            if same_types(self.resume_configuration(), cache['config']) \
                and id(input_data) == cache['input_id']:
                return cache['output']
        return None
    
    def add_cache(self, input_data:Input, output:Output) -> bool:
        """
        Add an input, output pair to cache

        Args:
            input_data (Input): Input to add
            output (Output): Result output to add

        Returns:
            bool: Success ? 
        """
        if not self.use_cache:
            return False
        
        self.caches.append({
            'input_id': id(input_data),
            'config': deepcopy(self.resume_configuration()),
            'output': output
        })
        return True
    
    def reset_cache(self) -> None:
        """
        Remove all cached data
        """
        self.caches = []
        
    @property
    def use_cache(self) -> bool:
        """Does cache is enable ?

        Returns:
            bool: True if enable
        """
        return self.__use_cache
    
    @use_cache.setter
    def use_cache(self, value:bool) -> bool:
        """
        Enable / Disable caching

        Args:
            value (bool): Value to set

        Raises:
            ValueError: Value must be a boolean

        Returns:
            bool: New use_cache value
        """
        if isinstance(value, bool):
            self.__use_cache = value
            return self.__use_cache
        
        raise ValueError('Value must be a boolean')
    
    
    
    def json_pipeline(self) -> dict:
        """
        Return a JSON formatted Step

        Returns:
            dict: Step in a json format
        """
        return {
            'step': self.__class__.__name__,
            'name': self.name,
            'description': self.description,
            'configuration': self.configurations[0],
            'children': []
        }
    

    def conf_to_rich_str_list(self) -> list:
        """
        Get configuration in a list of rich string format

        Returns:
            list: Configurations for rich logger
        """
        conf = [ f'{name}={conf["value"]}' for name, conf in self.configurations[0].items() ]
        
        return conf


    def to_rich_str(self) -> str:
        """
        Step in a rich formatted string

        Returns:
            str: Step description in a rich formatted string
        """
        conf = self.conf_to_rich_str_list()

        if len(conf) > 0:
            name = f'[b]{self.__class__.__name__}[/] ({", ".join(conf)})'
        else:
            name = f'[b]{self.__class__.__name__}[/]'
        
        return name
    
    def count_steps(self) -> int:
        """
        Returns a rough estimation of the total count of steps for a given
        pipeline.
        """
        return 1
    
    ############
    # Priorize #
    ############
    #
    # Base on the dataset (or not) priorize usefulness of this actionable.
    # Result is a value between 0 and 1. 0 stand for not useful
    #
    def priorize(self, input_data:Input=None) -> float: # pylint: disable=unused-argument
        """
        Try to evaluate the priorities level of himself on an input  

        Args:
            input_data (Input, optional): Input used to compute priorities. Defaults to None.

        Returns:
            float: Continuous value 0 to 1 
        """
        return 0.0
    
    #######
    # RUN #
    #######
    def run(self, input_data:'Input') -> Output:
        """
        Run the step on input data

        Args:
            input_data (Input): Input informations 

        Returns:
            Output: transformed Input 
        """
        return input_data.to_output()
    
    ###########
    ## STACK ##
    ###########
    # Stack all step ran to have a better understanding of pipeline execution. 
    # Each Step will store data in the stack. 
    # So we'll be able to unstack it and explain every data transformation in the pipeline
    
    def to_stack(self) -> Stack:
        """
        Transform a Step into Stack element 

        Returns:
            Stack: Stack representation of the step 
        """
        return Stack(
            self.__class__,
            self.current_configuration,
            id(self)
        )
        
    # Explain 
    @classmethod
    def explain(cls, config:dict) -> str:
        """
        Basic explanation of the Step
        """
        return f"""
            # {cls.name}
            {cls.description}
            {cls.__resume_a_configuration(config)}
        """
        
    # Track output
    def track_output(self, output:Output) -> None:
        """
        Automatically add Stack & call destroyers methods

        Args:
            output (Output): Output to track
        """
        if isinstance(self, Step):
            if type(output) in [Output, Input]:
                output.add_stack(self.to_stack())
            else:
                for one_output in output:
                    one_output.add_stack(self.to_stack())
        
        for destroyer in self.destroyers:
            destroyer.track_output(self, output)
    
    @classmethod
    def find_steps_by_tag(cls, tag:str) -> list['Step']:
        """
        Find all available Step with a specific Tag

        Args:
            tag (str): Tag to search for

        Returns:
            list[Step]: Found Steps
        """
        return set(filter(lambda key: tag in cls.available_steps[key], cls.available_steps.keys()))
    
    
#############   
# Decorator #
#############

#
# Class decorators
#

def is_step(*tags) -> callable:
    """
    is_step is needed to declare new Step.
    With the Step inheritance, it will setup everything to make it work smoothly
    Tags -> Your Step will be attached to these tags. 
        tags are use to easily include Step into Pipeline
    """
    def step_wrapper(cls) -> Step:
        """
        Declare the new step to Automed
        Add call to Step.__init__() so the Sub Step developer have one to care about this
        Returns:
            Step: Edited class
        """
        Step.available_steps[cls] = tags # Declare your Step to AutoMed
        
        # Help Python to find parent class
        __class__ = cls # pylint: disable=unused-variable
        
        
        initial_init = cls.__init__ # Keep the __init__ you have created
        def __init__(self, *args, **kw):
            if cls != Step:
                super().__init__(*args, **kw) # All parent constructor 
            
            initial_init(self, *args, **kw) # Run your __init__
            self.default_configurations() # Setup default configuration
            
        cls.__init__ = __init__ # Replace your init
            
        return cls
        
    return step_wrapper

#
# Method decorator
#

def runner(func) -> callable:
    """
    runner MUST decorate your run() method. It you manage every boring things for you.
        - Run configurations one by one
        - Store results in cache
        - Send information to Destroyers
        - Put results in good shape
        - Increment Stack data
        - Call callback method
        - And maybe more

    Args:
        func (callable): decorated method

    Returns:
        callable: edited method
    """
    def runner_wrapper(self, inputs:list[Input],
                        callback:callable=None
                        ) -> list[Output]:
        """Wrapping decorated method

        Returns:
            list[Output]: All generated outputs
        """
        
        if inputs.__class__ in [Output, Input]:
            inputs = [inputs]
        
        result:list[Output] = []
        for current in self.configurations:
            self.current_configuration = current

            # only print "parent" steps to reduce logs
            if hasattr(self, 'step') or hasattr(self, 'steps'):
                Logger().log(f'running step: {self.to_rich_str()}')
            
            for current_input in inputs:
                # Destroyer will stop Step run if results are not good enough
                # for destroyer in self.destroyers:
                #     if destroyer.destroyed(self):
                #         return result
                if self.suitable(current_input):
                    output = self.from_cache(current_input)
                    if not output:
                        output = func(self, current_input, callback=callback)
                        self.add_cache(current_input, output)
                    else:
                        callback(self) # Call callback manually because we used cache
                        
                    self.track_output(output)
                else:
                    output = current_input    
                
                    
                result = result + ([output] if type(output) in [Output, Input] else output)

        self.output = result        
        if callback:
            callback(self)
            
        return result
    return runner_wrapper


## Other methods
def same_types(a:dict, b:dict) -> bool:
    """
    Deep check of two dict. 
    Return true if both have same keys and values

    Args:
        a (dict): To compare with b
        b (dict): To compare with a

    Returns:
        bool: same types ?
    """
    if len(a.keys()) != len(b.keys()):
        return False
    for key, value in a.items():
        if isinstance(value, dict):
            return same_types(value, b[key])
        if key not in b or value != b[key]:
            return False
    return True
