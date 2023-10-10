from .output import Output, Input
from .dataset import Dataset
from .stack import Stack

from copy import deepcopy

# Step class is a brick used to create pipelines. This Step class is not really use in Pipeline, run function doesn't do anything. 
# This class is use to create new kinds of steps by inheritance and give all needed attributes and methods to children classes. 
# 
# There is also decorator needed to create a Step. See it under Step class.
#
class Step:
    # Available steps. This will be filled be all the new Step loaded in Python environements
    # It will be a reference of all available Steps to create pipeline
    available_steps = {}
    
    # Last output run of the Step
    output = None
    
    # Configuration of the Step. Each Step can of configuration and will save it here. Step give many method to help user to configure Steps
    configurations = [{}]
    current_configuration = [] # Current configuration (because a Step can have several)
    
    # Name and description of the Step. Useful to explain pipeline to users
    name = "Step" 
    description = "Step description..."
    
    # Will contain paper citations used to create this Step
    citations = [] # TODO
    
    # Last input of this Step. # TODO Still used ?
    input:Output = Output(None, None, None)
    
    def __init__(self, input:Output = Output(None, None, None), use_cache=True,  *args, **kw):
        self.__use_cache = use_cache # Activate or not the cache of results.
        self.caches = [] # Cached results
        
        self.destroyers = [] # List of destroyer. Destroyers are object with a global vision of the pipeline and are able to stop a pipeline branch if the results is badder then others branches
        self.parents_steps = [] # List all the previous steps before this one
        
        self.default_configurations() # Load default configuration 
        
        if input:
            self.input = input
        
    def __str__(self):
        return self.name
    
    @property
    def dataset(self):
        return self.input.dataset
    
    @property
    def metric(self):
        return self.input.metric
    
    @property
    def model(self):
        return self.input.model
    
    # When a Step contain others ones, this will help to setup everything (Transmit destroyers, increment parents steps)
    def configure_child(self, step, *args, **kw):
        child = step
        if any(self.destroyers):
            child.destroyers = self.destroyers
            
        child.parents_steps = self.parents_steps + [id(self)]
        
        return child
    
    ################
    # Configurable #
    ################
    #
    # A actionable step can be configured by the user.
    # For example, we can configure learning rate of a machine learning step.
    #  
    # Each parameters have a name, a description and a default value. Default value can be fixed or computed based on dataset
    
    # Configure one parameter 
    # config_id -> Index of the configuration
    # key -> Name of the parameter
    # value -> Value of the parameter 
    def configure_one(self, config_id, key, value):
        if key in self.configurations[config_id].keys():
            self.configurations[config_id][key]['value'] = value
        else:
            raise Exception(f"Configurable Key '{key}' does not exist.")
        
    # Configure all parameters with a ne dictionary
    def configure(self, dict, config_id=None):
        if config_id:
            for key, value in dict.items():
                self.configure_one(config_id, key, value)
        else:
            self.add_config(self, dict)
            
    # Add a fully new configuration
    def add_config(self, dict):
        pass
        # TODO
    
    # Resume a configuration -> No meta data, only "key: value"
    # config_id -> index of the configuration to resume, if None either the current_configuration or the first one will be choose
    def resume_configuration(self, config_id=None):
        if config_id:
            return Step.resume_a_configuration(self.configurations[config_id])
        else:
            return Step.resume_a_configuration(self.current_configuration or self.configurations[0])
    
    @classmethod
    def resume_a_configuration(cls, config):
        return {k: Step.__get_a_value(v) for k, v in config.items()}
    
    # Resume all configurations
    def resume_configurations(self):
        return {k: self._get_value(v) for k, v in self.configurations.items()}
    
    
    def _get_value(self, elem):
        return Step.__get_a_value(elem)
    
    @classmethod
    def __get_a_value(cls, elem):
        return elem['value'] if 'value' in elem.keys() else elem['default']
    
    # Get value of a configuration key. Very useful to easily get configuration in inherit Step methods
    def get_config(self, key):
        return self.resume_configuration()[key]
            
    # Defaults values of configuration
    def default_values(self, input=None):
        return {k: v['default'] for k, v in self.configurations.items()}
    
    def default_configurations(self):
        for index, current in enumerate(self.configurations):
            for key, elem in current.items():
                self.configurations[index][key]['value'] = self.configurations[index][key]['default']
                
    # Remove all configurations except the first one. Useful when you want a better control of step execution
    def keep_only_first_config(self):
        self.configurations = [self.configurations[0]]
        
    def all_configurations(self):
        return [{
            'step_id': id(self),
            'configuration': self.configurations
        }]
        
        
    #####################
    ## CACHING RESULTS ##
    #####################
    # Results of run() can by stored in cache to avoid compute it several time
    
    # If a previous run with same input & configuration was cached, return it
    # Else return False
    def from_cache(self, input):
        if not self.use_cache:
            return False
        
        for cache in self.caches:
            if same_types(self.resume_configuration(), cache['config']) and input == cache['input']:
                return cache['output']
        return False
    
    # Add an output to cache
    def add_cache(self, input, output):
        if not self.use_cache:
            return False
        
        return self.caches.append({
            'input': input,
            'config': deepcopy(self.resume_configuration()),
            'output': output
        })
    
    # Remove all cached data
    def reset_cache(self):
        self.caches = []
        
    @property
    def use_cache(self):
        return self.__use_cache
    
    @use_cache.setter
    def use_cache(self, value):
        if type(value) == bool:
            self.__use_cache = value
            return self.__use_cache
        else:
            raise Exception('Value must be a boolean')
    
    
    ############
    # Priorize #
    ############
    #
    # Base on the dataset (or not) priorize usefulness of this actionable.
    # Result is a value between 0 and 1. 0 stand for not useful
    #
    def priorize(self, input=None):
        return 0  
    
    ##########
    # Result #
    ##########
    def get_result(self):
        return self.output_dataset
    
    #######
    # RUN #
    #######
    #
    # Execute actionable on dataset with args (configurable and/or default_config). 
    #
    def run(self, dataset, *args):
        self.output_dataset = dataset
        return self.get_result()
    
    ###########
    ## STACK ##
    ###########
    # Stack all to have a better understanding of pipeline execution. 
    # Each Step will store data in the stack. So we'll be able to unstack it and explain every data transformation in the pipeline
    
    # Transform a Step into Stack element 
    def to_stack(self):
        return Stack(
            self.__class__,
            self.current_configuration,
            id(self)
        )
        
    # Explain 
    @classmethod
    def explain(cls, config):
        return f"""
            # {cls.name}
            {cls.description}
            {cls.resume_a_configuration(config)}
            
        """
        
    # Track output 
    # Automatically add Stack & call destroyers methods
    def track_output(self, output):
        if type(self) != Step:
            if type(output) in [Output, Input]:
                output.add_stack(self.to_stack())
            else:
                for one_ouput in output:
                    one_ouput.add_stack(self.to_stack())
        
        for destroyer in self.destroyers:
            destroyer.track_output(self, output)
    
    
#############   
# Decorator #
#############

#
# Class decorators
#

# isStep is needed to declare new Step. With the Step inheritance, it will setup everything to make it work smoothly
# Tags -> Your Step will be attached to these tags. 
# tags are use to easily include Step into Pipeline
def isStep(*tags):
    def stepWrapper(cls):
        Step.available_steps[cls] = tags # Declare your Step to AutoMed
        __class__ = cls # Help Python to find parent class
        
        
        initial_init = cls.__init__ # Keep the __init__ you have created
        def __init__(self, *args, **kw):
            if cls != Step:
                super().__init__(*args, **kw) # All parent constructor 
                
            initial_init(self, *args, **kw) # Run your __init__
            self.default_configurations() # Setup default configuration
            
        cls.__init__ = __init__ # Replace your init
            
        return cls
        
    return stepWrapper

# Will help AutoMed to know which Step is assessable (Learning step for example)
def assessable(cls): # Évaluable
    cls.metric = lambda output: 0 # Arbitrary metic
    cls.assessable = True
    
    def evaluate(self):
        return self.metric(self.output.dataset)
    
    def set_metric(self, metric):
        self.metric = metric
        
    cls.evaluate = evaluate
    cls.set_metric = set_metric
    
    return cls

#
# Method decorator
#

# runner MUST decorate your run() method. It you manage every boring things for you.
# - Run configurations one by one
# - Store results in cache
# - Send information to Destroyers
# - Put results in good shape
# - Increment Stack data
# - Call callback method
# - And maybe more
def runner(func):
    def runner_wrapper(self, inputs, callback=None, *args, **kw):
        
        if inputs.__class__ in [Output, Input]:
            inputs = [inputs]
        
        result:list[Output] = []
        for current in self.configurations:
            self.current_configuration = current
            print("# RUN #", self, self.resume_configuration())
            for input in inputs:
                
                # Destroyer will stop Step run if results are not good enough
                for destroyer in self.destroyers:
                    if destroyer.destroyed(self):
                        return result
            
                output = self.from_cache(input)
                if not output:
                    output = func(self, input, callback=callback, *args, **kw)
                    self.add_cache(input, output)
                    
                self.track_output(output)
                    
                result = result + ([output] if type(output) == Output else output)
                
        # print("->", result)
                    

        self.output = result        
        if callback:
            callback(self)
            
        return result
    return runner_wrapper


## Other methods
def same_types(a, b):
    if len(a.keys()) != len(b.keys()):
        return False
    for key, value in a.items():
        if isinstance(value, dict):
            return same_types(value, b[key])
        elif key not in b or value != b[key]:
            return False
    return True