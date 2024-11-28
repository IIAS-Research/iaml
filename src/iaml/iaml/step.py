"""
Step class is a brick used to create pipelines.
This class is not really use in Pipeline, run function doesn't do anything. 
Step class is use to create new kinds of steps by inheritance and give all
needed attributes and methods to children classes. 

There is also decorators needed to create a Step. See it under Step class.
"""
from __future__ import annotations

import sys
import json
from hashlib import md5
from typing import TYPE_CHECKING
from typing import Any, List, Dict
from copy import deepcopy
from multipledispatch import dispatch
from .dataset import Dataset
from .decorators.runner import runner
from .reference import Reference

if TYPE_CHECKING:
    from .candidate import Candidate
    from ..iaml.reference import Reference

class Step: # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """
    This class is the base for all IAML steps.
    
    Attributes:
        STATIC available_steps (dict) : Reference of all available Steps to create pipeline
        STATIC name (str) : Name of the step
        STATIC __description (str) : Description of the step
        candidate (Candidate): Last candidate of the Step
        configuration (dict) : Configuration of the step
        self.__use_cache (bool) : Enable / Disable caching
        caches (list) : Cached result 
        explanations (list[str]) : String explanation of step actions
        references (Reference) : List of references for this step
        citation (int, int) : String citations of step's references
    """
    # Available steps. This will be filled be all the new Step loaded in Python environments
    # It will be a reference of all available Steps to create pipeline
    available_steps = {}
    
    # Name and description of the Step. Useful to explain pipeline to users
    name = "Step" 
    refs = None
    __description = "Step description..."
    
    # By default a Step can be disabled. This attribute can by change to force a step to stay active
    can_be_disabled = True
    
    def __init__(self, *args, use_cache:bool=True, **kwargs): # pylint: disable=unused-argument
        """
        Initialize a step

        Parameters
        ----------
        use_cache : bool
            Do we use caching for this step
        """
        self.tags:set = None # Will be set by is_step
        self.__use_cache:bool = use_cache # Activate or not the cache of results.
        self.caches:list = [] # Cached results
        self.explanations:list[str] = []
        self.is_interchangeable:bool = False # Can be mutate into another step with the same tags
        self.enable:bool = True # A disable step, only take input and push it into output
        self.optimizable:bool = False # Does the parameters of this step is optimizable in stages ?

        self.configuration:dict = {}
        """Step give many method to help user to configure Steps
        Configuration of the Step. Each Step can have one configuration and will save it here."""
        
        self.parents_steps:list[Step] = [] # List all the previous steps before this one
        
        self.default_configuration() # Load default configuration 
        
        self.references:List[Reference] = []
        
        # Build a list of References from Step's references list
        if self.refs is not None:
            self.references = [Reference(ref, type(self).__name__) for ref in self.refs]

    @classmethod
    def from_pipeline(cls, pipeline: Dict, *args, **kwargs) -> 'Step':
        """
        Load any kind of Step (Step, MetaStep, Wrapper, etc) from json pipeline
        """
        step = None
        if 'step' not in pipeline:
            raise TypeError("invalid pipeline: missing step attribute")
        
        step_class = getattr(sys.modules['iaml'], pipeline['step']) # Get class from string
        if Step in step_class.__mro__:
            if step_class == cls:
                step = cls(*args, **kwargs)
                
                if 'enable' in pipeline:
                    step.enable = pipeline['enable']
            
                if 'configuration' in pipeline:
                    for name, value in pipeline['configuration'].items():
                        step.configure(name, value['value'])
            else: 
                step = step_class.from_pipeline(pipeline)
        else: 
            raise TypeError('invalid pipeline: step does not exist')
        
        return step
    
    @property
    def enable(self) -> bool:
        """
        Tells whether the step is enabled.

        :return: Whether the step is enabled
        """
        return self.__enable
    
    @enable.setter
    def enable(self, value: bool) -> bool:
        """
        Sets the state of the step.

        :param value: State of the step (True if enabled, False if disabled)
        """
        # If can_be_disabled = False -> Value will always be True
        self.__enable = value or not self.can_be_disabled

    def __str__(self):
        return self.name
    
    def suitable(self, dataset: Dataset) -> bool: # pylint: disable=unused-argument
        """
        Evaluates whether the step is suitable to run with the given dataset.

        :param dataset: Dataset to evaluate on
        :return: Whether the step is suitable to run
        """
        return True
    
    def configure_child(self, child: 'Step') -> 'Step':
        """
        When working with steps that can have children, setups the child with the parent.

        :param child: Step to configure with the parent
        :return: Configured step
        """
        child.configure_parents(self)

        return child

    def configure_parents(self, *parents) -> None:
        """
        Backpropagates the parents to the children.

        :param parents: Parent steps to add to the children
        """
        self.parents_steps.extend(map(id, parents))
        
    def step_with_same_tags(self) -> List['Step']:
        """
        Returns the steps that have the same tags as this step.

        :return: List of steps
        """
        return [ key for key, tags in Step.available_steps.items() if self.tags == set(tags) ]
    
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
        Configures a parameter.

        :param key: Name (or key) of the parameter to set
        :param value: New value for the parameter
        :raise AttributeError: When the key is invalid
        """
        if key in self.configuration:
            self.configuration[key]['value'] = value
        else:
            raise AttributeError(f"Configurable Key '{key}' does not exist.")
        
    @dispatch(dict)
    def configure(self, config: Dict[str, Any]): # pylint: disable=function-redefined
        """
        Configures several parameters at once.

        :param config: Dictionary of parameters' names and values
        """
        for key, value in config.items():
            self.configure(key, value)

    def passthrough_parameters(self, default: bool = True) -> Dict[str, Any]:
        """
        Constructs a dictionary of parameters' names and values.

        :param default: Default behavior when trying to passthrough configurations from one step
            which have no "passthrough" key. When "passthrough" is undefined and "default" is set
            to False, the configuration will not be returned; otherwise, the default value for that
            configuration will be returned.

        :return: A dictionary view of the configuration with parameters' names and values
        """
        parameters = {}
        for key, value in self.configuration.items():
            if "passthrough" in value:
                if value['passthrough']:
                    parameters[key] = value['value']
            elif default:
                parameters[key] = value['value']
                
        return parameters
            
    def all_configurations(self) -> List[Dict[str, Dict]]:
        """
        Retrieves the signature (id and configuration) of the step. This should be overloaded in
        steps which have children to also return the children's configuration.
        
        :return: List of all configurations as objects
        """
        return [{
            'step_id': id(self),
            'configuration': self.configuration
        }]

    def resume_configuration(self) -> Dict:
        """
        Resume a configuration -> No meta data, only "key: value"

        Returns
        -------
        Dict
            Configuration resume
        """
        return Step.__resume_a_configuration(self.configuration)
    
    def serializable_resume_configuration(self) -> dict:
        """
        Used by fingerprint methods
        
        Returns
        -------
        Dict
            Serializable configuration
        """
        return {key: value.__name__ if callable(value) else value \
            for key, value in self.resume_configuration().items()}
    
    @classmethod
    def __resume_a_configuration(cls, config: Dict) -> Dict:
        """
        Summarizes a configuration, and uses the default value for each configuration which have no
        value yet.

        :param config: Step's configuration
        :return: The configuration to make a "summary" for
        """
        return { k: Step.__get_a_value(v) for k, v in config.items() }
    
    @classmethod
    def __get_a_value(cls, elem: dict) -> Any:
        """
        Retrieves a value from the configuration, and uses the default value for that configuration
        if it has no value.

        :param elem: Step's configuration item
        :return: Value or default value
        """
        return elem['value'] if 'value' in elem.keys() else elem['default']
    
    def get_config(self, key: str) -> Any:
        """
        Retrieves 
        Get value of a configuration key.
        Very useful to easily get configuration in inherit Step methods

        Parameters
        ----------
        key : str
            Parameter to get

        Returns
        -------
        Any
            Value of parameter
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
            
    def check_configuration(self, fix: bool = True) -> bool:
        """Check if configuration is valid

        Parameters
        ----------
        fix : bool
            If True, invalid configuration will be fix. Defaults to True.

        Returns
        -------
        bool
            Is configuration valid ?
        """
        for key, config in self.configuration.items():
            if 'categorical' in config:
                if self.get_config(key) not in config['categorical']:
                    if fix:
                        self.configure(key, config['categorical'][0])
                    else:
                        return False
            elif 'range' in config:
                if self.get_config(key) < config['range'][0] \
                    or self.get_config(key) > config['range'][1]:
                    if fix:
                        self.configure(key, (config['range'][0]+config['range'][1])/2)
                    else:
                        return False
        return True
    
    def all_steps(self) -> List:
        """Recursive function (last one here) to get all steps in a pipeline

        Returns
        -------
        List
            always empty
        """
        return [self]
        
        
    #####################
    ## CACHING RESULTS ##
    #####################
    # Results of run() can by stored in cache to avoid compute it several time
    
    def from_cache(self, candidate: Candidate) -> Candidate:
        """
        If a previous run with same candidate & configuration was cached, return it
        Else return None

        Parameters
        ----------
        candidate : Candidate
            Try to find this candidate in cache

        Returns
        -------
        Candidate
            Cached candidate or None
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

        Parameters
        ----------
        input_candidate : Candidate
            Candidate to add
        output_candidate : Candidate
            Result candidate to add

        Returns
        -------
        bool
            Success ? 
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

        Returns
        -------
        bool
            True if enable
        """
        return self.__use_cache
    
    @use_cache.setter
    def use_cache(self, value:bool) -> bool:
        """
        Enable / Disable caching

        Parameters
        ----------
        value : bool
            Value to set

        Raises
        ------
        ValueError
            Value must be a boolean

        Returns
        -------
        bool
            New use_cache value
        """
        if isinstance(value, bool):
            self.__use_cache = value
            return self.__use_cache
        
        raise ValueError('Value must be a boolean')
    
    
    
    def json_pipeline(self) -> Dict[str, Any]:
        """
        Return a JSON formatted Step

        Returns
        -------
        Dict[str, Any] 
            Step in a json format
        """
        return {
            'step': self.__class__.__name__,
            'name': self.name,
            'description': self.description,
            'enable': self.enable,
            'can_be_disable': self.can_be_disabled,
            'configuration': self.configuration,
            'children': []
        }

    def conf_to_rich_str_list(self) -> List[str]:
        """
        Get configuration in a list of rich string format

        Returns
        -------
        List[str]
            Configurations for rich logger
        """
        conf = [ f'{name}={conf["value"]}' for name, conf in self.configuration.items() ]
        
        return conf


    def to_rich_str(self) -> str:
        """
        Step in a rich formatted string

        Returns
        -------
        str
            Step description in a rich formatted string
        """
        conf = self.conf_to_rich_str_list()

        if len(conf) > 0:
            name = f'[b]{self.__class__.__name__}[/] ({", ".join(conf)})'
        else:
            name = f'[b]{self.__class__.__name__}[/]'
        
        return name
    
    def count_steps(self) -> int:
        """
        Returns a rough estimation of the total count of steps for a given pipeline.
        
        :return: Number of steps
        """
        return 1
    
    ############
    # Priorize #
    ############
    def priorize(self, candidate: Candidate = None) -> float: # pylint: disable=unused-argument
        """
        Evaluates the priority of the step within a pipeline. 0 means the execution of the step
        should not be prioritized, and 1 means it should be executed early in the pipeline.

        :param Candidate, optional candidate: Candidate on which the priority should be evaluated.
        :return: Value between 0 and 1.
        """
        return 0
    
    #######
    # RUN #
    #######
    def fit(self, dataset: Dataset) -> 'Step':  # pylint: disable=unused-argument
        """
        Fit the step on the given dataset.

        :param dataset: Features and labels
        :return: Fitted step
        """
        return self
    
    @runner  
    def run(self, candidate: Candidate) -> Candidate:
        """
        Runs the step for a given candidate.

        :param candidate: Candidate to run the step for
        :return: Run candidate 
        """
        self.fit(candidate.dataset)

        return candidate.add_to_pipeline(self)
    
    @classmethod
    def find_steps_by_tag(cls, tag:str) -> List['Step']:
        """
        Find all available Step with a specific Tag

        Parameters
        ----------
        tag : str
            Tag to search for

        Returns
        -------
        List[Step]
            Found Steps
        """
        return set(filter(lambda key: tag in cls.available_steps[key], cls.available_steps.keys()))
    
    
    def fingerprint(self) -> str:
        """
        Return a md5 hash that can by use to compare Step 

        Returns
        -------
        str
            md5 sting
        """
        to_hash = f"{str(self.__class__)} = \
            {json.dumps(self.serializable_resume_configuration(), sort_keys=True)}"
        return md5(to_hash.encode()).hexdigest()
    
    ####################
    ### Explanations ###
    ####################
    @property
    def description(self) -> str:
        """
        Formats the description of a step with its configuration.

        Returns
        -------
        str
            Formatted description.
        """
        conf = { k: v['value'] for k, v in self.configuration.items() }

        return self.__description.format(**conf)

    @description.setter
    def description(self, value: str) -> str:
        """
        Description setter
        
        Parameters
        ----------
        value : str
            The new description
        
        Returns
        -------
        str
            The new description
        """
        self.__description = value
        return self.description

    def explain(self, explanations_limit: int = 20) -> str:
        """
        Renders the explanation as Markdown text.

        Parameters
        ----------
        explanations_limit : int
            Maximum number of explanations to provide

        Returns
        -------
        str
            Markdown text.
        """
        # if not self.explanations:
        #     return ''
        explanations = '\n'.join([ f' - {p}' for p in self.explanations[:explanations_limit] ])

        if len(explanations) == 0:
            return None

        confs = '\n'.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'
            for k, v in self.configuration.items()
        ])

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
def same_types(a: Dict, b: Dict) -> bool:
    """
    Deep check of two dict. 
    Return true if both have same keys and values

    Parameters
    ----------
    a : Dict
        To compare with b
    b : Dict
        To compare with a

    Returns
    -------
    bool
        same types ?
    """
    if len(a.keys()) != len(b.keys()):
        return False
    for key, value in a.items():
        if isinstance(value, dict):
            return same_types(value, b[key])
        if key not in b or value != b[key]:
            return False
    return True
