"""
Step class is a brick used to create pipelines.
This class is not really use in Pipeline, run function doesn't do anything. 
Step class is use to create new kinds of steps by inheritance and give all
needed attributes and methods to children classes. 

There is also decorators needed to create a Step. See it under Step class.
"""

import sys
from typing import Any
from copy import deepcopy
from multipledispatch import dispatch
from .candidate import Candidate
from .dataset import Dataset
from .stack import Stack
from .decorators.runner import runner

class Step: # pylint: disable=too-many-public-methods
    """
    Step class is use to create new kinds of steps by inheritance and give all needed
    attributes and methods to children classes. 
    
    Attributes:
        STATIC available_steps (dict) : Reference of all available Steps to create pipeline
        STATIC name (str) : Name of the step
        STATIC __description (str) : Description of the step
        candidate (Candidate): Last candidate of the Step
        configuration (dict) : Configuration of the step
        self.__use_cache (bool) : Enable / Disable caching
        caches (list) : Cached result 
        explanations (list[str]) : String explanation of step actions
    """
    # Available steps. This will be filled be all the new Step loaded in Python environments
    # It will be a reference of all available Steps to create pipeline
    available_steps = {}
    
    
    # Name and description of the Step. Useful to explain pipeline to users
    name = "Step" 
    __description = "Step description..."
    
    def __init__(self, *args, use_cache:bool=True, **kwargs): # pylint: disable=unused-argument
        self.candidate:Candidate = None
        self.__use_cache:bool = use_cache # Activate or not the cache of results.
        self.caches:list = [] # Cached results
        self.explanations:list[str] = []
        
        # Configuration of the Step. Each Step can have one configuration and will save it here.
        # Step give many method to help user to configure Steps
        self.configuration:dict = {}
        
        self.parents_steps:list[Step] = [] # List all the previous steps before this one
        
        self.default_configuration() # Load default configuration 
        
        
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
                        step.configure(name, value['value'])
            else: 
                step = step_class.from_pipeline(pipeline)
        else: 
            raise TypeError('invalid pipeline: step does not exist')
        
        return step
    
    # def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None: # pylint: disable=unused-argument
    #     """
    #     WIll always raise NotImplementedError.

    #     Args:
    #         X (pd.DataFrame): X Data
    #         y (pd.DataFrame): Y data

    #     Raises:
    #         NotImplementedError: AutoMed Step can't be fit this way. You have to use AutoMed.run()
    #     """
    #     raise NotImplementedError("AutoMed Step can't be fit. You have to use AutoMed.run()")
        
    def __str__(self):
        return self.name
    
    def suitable(self, candidate:Candidate) -> bool: # pylint: disable=unused-argument
        """
        Have to be overwrote. Check if a step is suitable for a given Candidate

        Args:
            candidate (Candidate): Candidate to test

        Returns:
            bool: Is it suitable ?
        """
        return True
    
    def configure_child(self, child:'Step') -> 'Step':
        """
        When a Step contain others ones, this will help to setup everything
        (increment parents steps)

        Args:
            child (Step): Step to configure

        Returns:
            Step: Configured step
        """

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
    
    @dispatch(str, object)
    def configure(self, key:str, value:Any) -> None:
        """
        Configure one parameter

        Args:
            key (str): Name of the parameter
            value (any): Value to set 

        Raises:
            Exception: _description_
        """
        if key in self.configuration:
            self.configuration[key]['value'] = value
        else:
            raise AttributeError(f"Configurable Key '{key}' does not exist.")
        
    @dispatch(dict)
    def configure(self, config:dict): # pylint: disable=function-redefined
        """
        Configure several parameters with a dictionary

        Args:
            config (dict): key as parameter name, value as value to set
        """
        for key, value in config.items():
            self.configure(key, value)
            
    def all_configurations(self) -> list[dict]:
        """Recursive function (last one here) to get all configurations in a pipeline

        Returns:
            list[dict]: list of all configurations
        """
        return [{
            'step_id': id(self),
            'configuration': self.configuration
        }]
    
    def resume_configuration(self) -> dict:
        """
        Resume a configuration -> No meta data, only "key: value"

        Returns:
            dict: Configuration resume
        """
        return Step.__resume_a_configuration(self.configuration)
    
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
        param:dict = self.configuration[key]
        return param['value'] if 'value' in param else param['default']
    
    def default_configuration(self) -> None:
        """
        Set default configuration
        Explore configuration parameters and set default as value
        """
        for param in self.configuration.values():
            param['value'] = param['default']
    
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
    
    def from_cache(self, candidate:Candidate) -> Candidate:
        """
        If a previous run with same candidate & configuration was cached, return it
        Else return None

        Args:
            candidate (Candidate): Try to find this candidate in cache

        Returns:
            Candidate: Cached candidate or None
        """
        if not self.use_cache:
            return None
        
        for cache in self.caches:
            if same_types(self.resume_configuration(), cache['config']) \
                and id(candidate) == cache['input_id']:
                return cache['output']
        return None
    
    def add_cache(self, input_candidate:Candidate, output_candidate:Candidate) -> bool:
        """
        Add an candidate, candidate pair to cache

        Args:
            candidate (Candidate): Candidate to add
            candidate (Candidate): Result candidate to add

        Returns:
            bool: Success ? 
        """
        if not self.use_cache:
            return False
        
        self.caches.append({
            'input_id': id(input_candidate),
            'config': deepcopy(self.resume_configuration()),
            'output': output_candidate
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
            'configuration': self.configuration,
            'children': []
        }
    

    def conf_to_rich_str_list(self) -> list:
        """
        Get configuration in a list of rich string format

        Returns:
            list: Configurations for rich logger
        """
        conf = [ f'{name}={conf["value"]}' for name, conf in self.configuration.items() ]
        
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
    def priorize(self, candidate:Candidate=None) -> float: # pylint: disable=unused-argument
        """
        Try to evaluate the priorities level of himself on an candidate  

        Args:
            candidate (Candidate, optional): Candidate used to compute priorities. Defaults to None.

        Returns:
            float: Continuous value 0 to 1 
        """
        return 0.0
    
    #######
    # RUN #
    #######
    def fit(self, dataset:Dataset) -> 'Step':
        """
        Fit Step on a Dataset.

        Args:
            dataset (Dataset): Features and labels

        Returns:
            Step: fitted step 
        """
        return self
    
    @runner  
    def run(self, candidate:'Candidate', callback:callable=None) -> Candidate:
        """
        Run the step on candidate data

        Args:
            candidate (Candidate): Candidate informations 

        Returns:
            Candidate: transformed Candidate 
        """
        self.fit(candidate.dataset)
        return candidate.add_to_pipeline(self)
    
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
            self.configuration,
            id(self)
        )
        
    # Track candidate
    def track_candidate(self, candidate:Candidate) -> None:
        """
        Automatically add Stack

        Args:
            candidate (Candidate): Candidate to track
        """
        if isinstance(self, Step):
            if type(candidate) in [Candidate]:
                candidate.add_stack(self.to_stack())
            else:
                for one_candidate in candidate:
                    one_candidate.add_stack(self.to_stack())
        
    
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
    
    
    ####################
    ### Explanations ###
    ####################
    @property
    def description(self) -> str:
        """
        Formats the description of a step with its configuration.

        Returns:
            str: Formatted description.
        """
        conf = { k: v['value'] for k, v in self.configuration.items() }

        return self.__description.format(**conf)

    @description.setter
    def description(self, value:str) -> str:
        """
        Description setter
        """
        self.__description = value
        return self.description
    
    def explain(self, explanations_limit: int = 20) -> str:
        """
        Renders the explanation as Markdown text.

        Returns:
            str: Markdown text.
        """
        if not self.explanations:
            return ''

        confs = '\n'.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'
            for k, v in self.configuration.items()
        ])

        explanations = '\n'.join([ f' - {p}' for p in self.explanations[:explanations_limit] ])

        explanations_left = len(self.explanations) - explanations_limit

        return f"""
## {self.name}
**{self.description}**

{f'''
### Configuration
| Name | Description | Value |
| ---- | ----------- | ----- |
{confs}
''' if len(confs) > 0 else ""}

{f'''
### Processings
{explanations}
{f" - *and **{explanations_left}** more explanations...*" if explanations_left > 0 else ""}
''' if len(explanations) > 0 else ""}
        """
        

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
