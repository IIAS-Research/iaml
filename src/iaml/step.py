"""
Step class is a brick used to create pipelines.
This class is not really use in Pipeline, run function doesn't do anything. 
Step class is use to create new kinds of steps by inheritance and give all
needed attributes and methods to children classes. 

There are also decorators needed to create a Step. See it under Step class.
"""

from __future__ import annotations

import sys
import json
import textwrap
import uuid
from hashlib import md5
from typing import TYPE_CHECKING
from typing import Any
from copy import deepcopy
from multipledispatch import dispatch
from .dataset import Dataset
from .decorators.runner import runner
from .reference import Reference
from .step_cache import StepCache

if TYPE_CHECKING:
    from .candidate import Candidate
    from ..iaml.reference import Reference


class Step: # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """This class is the base for all IAML steps.

    :param bool, optional use_cache: Whether to enable caching for this step.
    """
    available_steps: dict[Step, tuple[str]] = {}
    """Static list of all registered steps within IAML."""

    name: str = 'Step'
    """Step's name."""

    refs: list[dict[str, Any]] = None
    """List of references for this step."""

    _description: str = ''
    """Short description of the step."""

    _description_long: str = ''
    """Longer description of the step."""

    _usage: str = ''
    """Concise guidance on when to use (and avoid) this step."""

    can_be_disabled: bool = True
    """Whether this step can be disabled."""

    def __init__(self, *args, use_cache: bool = True, **kwargs): # pylint: disable=unused-argument
        self.tags: set = None
        """Step's tags. Tags are equivalent to categories of steps."""

        self.__use_cache: bool = use_cache
        """Whether this step should benefit from the cache."""

        self._cache_id: str = uuid.uuid4().hex
        """Shared cache namespace id for this step across deep copies."""

        self._config_version: int = 0
        """Incremented when configuration changes to invalidate fingerprints."""

        self.explanations: list[str] = []
        """List of explanations that were computed during the step's execution."""

        self.is_interchangeable: bool = False
        """Whether this step can be mutated into another step with the same tags."""

        self.enable: bool = True
        """Whether this step is enabled. If it is disabled, the step will simply return the input
        candidate."""

        self.optimizable: bool = False
        """Whether the configuration's parameters should be considered optimizable."""

        self.configuration: dict = {}
        """Configuration of this step (equivalent to hyperparameters for models)."""

        self.parents_steps: list[Step] = []
        """If this step is a child of another one, lists all the parents of this step."""

        self.references: list[Reference] = []
        """References for this step."""

        # Build a list of References from Step's references list
        if self.refs is not None:
            self.references = [ Reference(ref, type(self).__name__) for ref in self.refs ]

        self.default_configuration() # Loads the default configuration

    def __deepcopy__(self, memo: dict) -> 'Step':
        """Custom deepcopy to avoid copying per-step runtime caches."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == 'candidate':
                setattr(result, key, None)
                continue
            if key == 'caches':
                setattr(result, key, [])
                continue
            setattr(result, key, deepcopy(value, memo))
        return result

    @classmethod
    def from_pipeline(cls, pipeline: dict[str, Any], *args, **kwargs) -> 'Step':
        """Loads any kind of Step (Step, MetaStep, Wrapper, etc.) from an imported pipeline.

        :param dict pipeline: Pipeline to import.
        :param optional \\*args: Args to pass to the Step's constructor.
        :param optional \\**kwargs: Kwargs to pass to the Step's constructor.
        :raise TypeError: invalid pipeline: missing step attribute
        :raise TypeError: invalid pipeline: step does not exist
        :return: A step
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
                        step.configure(name, value['value']) # pylint: disable=too-many-function-args
            else:
                step = step_class.from_pipeline(pipeline)
        else:
            raise TypeError('invalid pipeline: step does not exist')

        return step

    @property
    def enable(self) -> bool:
        """Tells whether the step is enabled.

        :return: Whether the step is enabled.
        """
        return self.__enable

    @enable.setter
    def enable(self, value: bool) -> None:
        """Sets the state of the step.

        :param value: State of the step (True if enabled, False if disabled).
        """
        # If can_be_disabled is False, value will always be True
        self.__enable = value or not self.can_be_disabled

    def __str__(self) -> str:
        """Returns the name of this step.

        :return: Name of this step.
        """
        return self.name

    def suitable(self, dataset: Dataset) -> bool: # pylint: disable=unused-argument
        """Evaluates whether the step is suitable to run with the given dataset.

        :param Dataset dataset: Dataset to evaluate on.
        :return: Whether the step is suitable to run.
        """
        return True

    def configure_child(self, child: 'Step') -> 'Step':
        """When working with steps that can have children, setups the child with the parent.

        :param Step child: Step to configure with the parent.
        :return: Configured step.
        """
        child.configure_parents(self)

        return child

    def configure_parents(self, *parents: list['Step']) -> None:
        """Backpropagates the parents to the children.

        :param list[Step] parents: Parent steps to add to the children.
        """
        self.parents_steps.extend(map(id, parents))

    def step_with_same_tags(self) -> list['Step']:
        """Returns the steps that have the same tags as this step.

        :return: List of steps.
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
    def configure(self, key: str, value: Any) -> None:
        """Configures a parameter.

        :param str key: Name (or key) of the parameter to set.
        :param Any value: New value for the parameter.
        :raise AttributeError: When the key is invalid.
        """
        if key in self.configuration:
            self.configuration[key]['value'] = value
            self._config_version += 1
        else:
            raise AttributeError(f"Configurable Key '{key}' does not exist.")

    @dispatch(dict)
    def configure(self, config: dict[str, Any]) -> None: # pylint: disable=function-redefined
        """Configures several parameters at once.

        :param dict[str, Any] config: Dictionary of parameters' names and values.
        """
        for key, value in config.items():
            self.configure(key, value) # pylint: disable=too-many-function-args

    def passthrough_parameters(self, default: bool = True) -> dict[str, Any]:
        """Constructs a dictionary of parameters' names and values.

        :param bool default: Default behavior when trying to passthrough configurations from
            one step which have no "passthrough" key. When "passthrough" is undefined and "default" 
            is set to False, the configuration will not be returned; otherwise, the default value 
            for that configuration will be returned.
        :return: A dictionary view of the configuration with parameters' names and values.
        """
        parameters = {}
        for key, value in self.configuration.items():
            if "passthrough" in value:
                if value['passthrough']:
                    parameters[key] = value['value']
            elif default:
                parameters[key] = value['value']

        return parameters

    def all_configurations(self) -> list[dict[str, Any]]:
        """Retrieves the signature (id and configuration) of the step. This should be overridden in
        steps which have children to also return the children's configuration.

        :return: List of all configurations as objects.
        """
        return [{
            'step_id': id(self),
            'configuration': self.configuration
        }]

    def resume_configuration(self) -> dict:
        """Summarizes the configuration of this step as keys and values.

        :return: Configuration summary.
        """
        return Step.__resume_a_configuration(self.configuration)

    def serializable_resume_configuration(self) -> dict:
        """Generates a serializable configuration of this step.

        :return: Serializable dictionary.
        """
        return {key: value.__name__ if callable(value) else value \
            for key, value in self.resume_configuration().items()}

    @classmethod
    def __resume_a_configuration(cls, config: dict[str, Any]) -> dict:
        """Summarizes a configuration, and uses the default value for each configuration which have
        no value yet.

        :param dict[str, Any] config: Step's configuration.
        :return: The configuration to make a "summary" for.
        """
        return { k: Step.__get_a_value(v) for k, v in config.items() }

    @classmethod
    def __get_a_value(cls, elem: dict[str, Any]) -> Any:
        """Retrieves a value from the configuration, and uses the default value for that
        configuration if it has no value.

        :param dict[str, Any] elem: Step's configuration item.
        :return: Value or default value.
        """
        return elem['value'] if 'value' in elem.keys() else elem['default']

    def get_config(self, key: str) -> Any:
        """Retrieves a value from the configuration, and uses the default value for that
        configuration if it has no value.

        :param str key: Step's configuration item.
        :return: Value or default value.
        """
        param: dict = self.configuration[key]

        return param['value'] if 'value' in param else param['default']

    def default_configuration(self) -> None:
        """Gives the default value to all configuration items."""
        for param in self.configuration.values():
            param['value'] = param['default']

    def check_configuration(self, fix: bool = True) -> bool:
        """Checks whether the configuration's format is valid.

        :param bool fix: Whether to try to fix the configuration if something is wrong.
        :return: Whether the configuration is wrong. Fixed issues return True.
        """
        for key, config in self.configuration.items():
            if 'categorical' in config:
                if self.get_config(key) not in config['categorical']:
                    if fix:
                        self.configure(key, config['categorical'][0]) # pylint: disable=too-many-function-args
                    else:
                        return False
            elif 'range' in config:
                if self.get_config(key) < config['range'][0] \
                    or self.get_config(key) > config['range'][1]:
                    if fix:
                        self.configure(key, (config['range'][0] + config['range'][1]) / 2) # pylint: disable=too-many-function-args
                    else:
                        return False
        return True

    def all_steps(self) -> list[Step]:
        """Retrieves all steps of a pipeline. This should be overridden in steps which may run
        other steps, such as children steps.

        :return: List of steps.
        """
        return [self]
    #####################
    ## CACHING RESULTS ##
    #####################
    # Results of run() can be stored in cache to avoid recomputing the same candidate.

    def from_cache(self, candidate: Candidate) -> Candidate:
        """Checks whether this step was already run with the same input candidate and the same
        configuration, and return the cached result if that is the case. Otherwise, return None.

        :param Candidate candidate: Candidate to look for in cache.
        :return: Cached candidate or None.
        """
        if not self.use_cache:
            return None

        cache_key = self._cache_key(candidate)
        cached = StepCache().get(cache_key, candidate)
        if cached is None:
            return None

        return self._clone_output(cached)

    def add_cache(self, input_candidate: Candidate, output_candidate: Candidate) -> bool:
        """Adds a candidate in the cache of this step.

        :param Candidate input_candidate: Input candidate.
        :param Candidate output_candidate: Output candidate to cache.
        :return: Whether the candidate was cached.
        """
        if not self.use_cache:
            return False

        cache_key = self._cache_key(input_candidate)
        frozen_output = self._clone_output(output_candidate)
        StepCache().put(cache_key, frozen_output, self._cache_id, input_candidate)

        return True

    def reset_cache(self) -> None:
        """Removes all cached candidates from the cache."""
        StepCache().clear(self._cache_id)

    @property
    def caches(self) -> list:
        """Deprecated view of cached results backed by StepCache."""
        return StepCache().values_for_step(self._cache_id)

    @caches.setter
    def caches(self, value: list | None) -> None:
        if not value:
            StepCache().clear(self._cache_id)

    def _cache_key(self, candidate: Candidate) -> tuple:
        """Index by address; StepCache also verifies the input through a weak reference."""
        candidate_id = id(candidate) if candidate is not None else None
        return (self._cache_id, self.fingerprint(), candidate_id)

    def _clone_output(self, output: Any) -> Any:
        """Return cached output without cloning to preserve identity semantics."""
        return output

    @property
    def use_cache(self) -> bool:
        """Tells whether the step is using the cache.

        :return: True if caching is enabled, False otherwise.
        """
        return self.__use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> bool:
        """Enables or disables caching.
        
        :param bool value: If True, enables caching; if False, disables caching.
        :raise ValueError: When `value` is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError('Value must be a boolean')
        self.__use_cache = value

    def json_pipeline(self) -> dict[str, Any]:
        """Exports a representation of this step as a dictionary.

        :return: Dictionary view of this step's signature.
        """
        return {
            'step': self.__class__.__name__,
            'name': self.name,
            'description': self.description,
            'enable': self.enable,
            'can_be_disable': self.can_be_disabled,
            'configuration': {
                k: { **v, 'description': v['description'].replace('\n', ' ') }
                for k, v in self.configuration.items() },
            'children': []
        }

    def conf_to_rich_str_list(self) -> list[str]:
        """Renders the configuration of this step as a list of rich-formatted strings.

        :return: List of rich-formatted strings.
        """
        conf = [ f'{name}={conf["value"]}' for name, conf in self.configuration.items() ]

        return conf

    def to_rich_str(self) -> str:
        """Renders the signature of this step as a rich-formatted string.

        :return: Rich-formatted string.
        """
        conf = self.conf_to_rich_str_list()

        if len(conf) > 0:
            name = f'[b]{self.__class__.__name__}[/] ({", ".join(conf)})'
        else:
            name = f'[b]{self.__class__.__name__}[/]'

        return name

    def count_steps(self) -> int:
        """Returns a rough estimation of the total count of steps for a given pipeline.
        
        :return: Number of steps.
        """
        return 1

    ############
    # Priorize #
    ############
    def priorize(self, candidate: Candidate = None) -> float: # pylint: disable=unused-argument
        """Evaluates the priority of the step within a pipeline. 0 means the execution of the step
        should not be prioritized, and 1 means it should be executed early in the pipeline.

        :param Candidate, optional candidate: Candidate on which the priority should be evaluated.
        :return: Value between 0 and 1.
        """
        return 0.0

    #######
    # RUN #
    #######
    def fit(self, dataset: Dataset) -> 'Step':  # pylint: disable=unused-argument
        """Fits the step on the given dataset.

        :param Dataset dataset: Features and labels.
        :return: Fitted step.
        """
        return self

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        """Runs the step for a given candidate.

        :param Candidate candidate: Candidate to run the step for.
        :return: Run candidate.
        """
        
        # Never fit predictor during generation of candidates
        if self.tags and 'predictor' in self.tags:
            return candidate.add_to_pipeline(self)

        self.fit(candidate.dataset)

        return candidate.add_to_pipeline(self)

    def fingerprint(self) -> str:
        """Returns a MD5 hash that can be used to distinguish steps.

        :return: MD5 string.
        """
        to_hash = f"{str(self.__class__)} = \
            {json.dumps(self.serializable_resume_configuration(), sort_keys=True)}"

        return md5(to_hash.encode()).hexdigest()

    ####################
    ### Explanations ###
    ####################
    def __format_description(self, description: str) -> str:
        return description \
            .replace('\n', ' ') \
            .format(**{ k: v['value'] for k, v in self.configuration.items() })

    @property
    def description(self) -> str:
        """Formats the description of a step with its configuration.

        :return: Formatted description.
        """
        return self.__format_description(self._description)

    @property
    def description_long(self) -> str:
        """Formats the longer description of a step with its configuration.
        
        :return: Formatted description.
        """
        return self.__format_description(self._description_long)

    def explain(self, processings_limit: int = 20) -> str:
        """Renders the step configurations as Markdown text.

        :param int processings_limit: Maximum number of processings to render.
        :return: Markdown document.
        """
        confs = '\n            '.join([
            f'| **{k}** | {v["description"]} | {v["value"]} |'.replace('\n', '')
            for k, v in self.configuration.items()
        ])

        processings = [ f' - {p}' for p in self.explanations[:processings_limit] ]
        explanations = '\n            '.join(processings)
        processings_left = len(self.explanations) - processings_limit

        if len(explanations) == 0 and (hasattr(self, 'transform') or hasattr(self, 'resample')):
            return None

        markdown_conf = textwrap.dedent(f"""\
            ### Configuration
            | Name | Description | Value |
            | ---- | ----------- | ----- |
            {confs}
            """) if len(confs) > 0 else ""

        markdown_processings = textwrap.dedent(f'''\
            ### Processings
            {explanations}
            {f" - *and **{processings_left}** more explanations...*" if processings_left > 0 else ""}
            ''') if len(explanations) > 0 else ""

        return '\n'.join([
            f'## {self.name}',
            f'**{self.description}**\n',                               
            markdown_conf,
            markdown_processings,
        ])

def same_types(a: dict, b: dict) -> bool:
    """Recursively checks whether the two provided dictionaries are the exact same.
    
    :param dict a: First dictionary.
    :param dict b: Second dictionary.
    :return: Whether the two dictionaries are the same.
    """
    if len(a.keys()) != len(b.keys()):
        return False

    for key, value in a.items():
        if isinstance(value, dict):
            return same_types(value, b[key])

        if key not in b or value != b[key]:
            return False

    return True
