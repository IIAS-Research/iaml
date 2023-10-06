from .output import Output
from .dataset import Dataset

from copy import deepcopy

class Step:
    # Available steps
    available_steps = {}
    output = None
    configurations = [{}]
    current_configuration = []
    name = "Step"
    
    input:Output = Output(None, None, None)
    
    def __init__(self, input:Output = Output(None, None, None)):
        self.caches = []
        self.default_configurations()
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
    
    ################
    # Configurable #
    ################
    #
    # A actionable step can be configured by the user.
    # For example, we can configure learning rate of a machine learning step.
    #  
    # Each parameters have a name, a description and a default value. Default value can be fixed or computed based on dataset
    
    def configure_one(self, config_id, key, value):
        if key in self.configurations[config_id].keys():
            self.configurations[config_id][key]['value'] = value
        else:
            raise Exception(f"Configurable Key '{key}' does not exist.")
        
    def configure(self, dict, config_id=None):
        if config_id:
            for key, value in dict.items():
                self.configure_one(config_id, key, value)
        else:
            self.add_config(self, dict)
            
    def add_config(self, dict):
        pass
        # TODO
    
    def resume_configuration(self, config_id=None):
        if config_id:
            return {k: self._get_value(v) for k, v in self.configurations[config_id].items()}
        else:
            return {k: self._get_value(v) for k, v in (self.current_configuration or self.configurations[0]).items()}
    
    def resume_configurations(self):
        return {k: self._get_value(v) for k, v in self.configurations.items()}
    
    def _get_value(self, elem):
        return elem['value'] if 'value' in elem.keys() else elem['default']
    
    def get_config(self, key):
        return self.resume_configuration()[key]
            
    
    def default_values(self, input=None):
        return {k: v['default'] for k, v in self.configurations.items()}
    
    
    def default_configurations(self):
        for index, current in enumerate(self.configurations):
            for key, elem in current.items():
                self.configurations[index][key]['value'] = self.configurations[index][key]['default']
                
    def keep_only_first_config(self):
        self.configurations = [self.configurations[0]]
        
        
    #####################
    ## CACHING RESULTS ##
    #####################
    # TODO -> Visiblement cela filtre un peu trop (baisse de résultats)
    def from_cache(self, input):
        for cache in self.caches:
            if same_types(self.resume_configuration(), cache['config']) and input == cache['input']:
                # print("CACHE USAGE !")
                # print("current ", self.resume_configuration())
                # print("cache ", cache['config'])
                # print("step", self)
                return cache['output']
        return False
    
    def add_cache(self, input, output):
        # print("CACHING !", self, self.current_configuration)
        return self.caches.append({
            'input': input,
            'config': deepcopy(self.resume_configuration()),
            'output': output
        })
    
    
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
    
    
    
#############   
# Decorator #
#############

#
# Class decorators
#

def isStep(*tags):
    def stepWrapper(cls):
        Step.available_steps[cls] = tags
        
        initial_init = cls.__init__
        def __init__(self, *args, **kw):
            initial_init(self, *args, **kw)
            Step.__init__(self)
            # self.default_configurations()
            
        cls.__init__ = __init__
            
        return cls
        
    return stepWrapper

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
def runner(func):
    def runner_wrapper(self, inputs, callback=None, *args, **kw):
        
        if inputs.__class__ == Output:
            inputs = [inputs]
        
        result:list[Output] = []
        for current in self.configurations:
            self.current_configuration = current
            print("# RUN #", self, self.resume_configuration())
            for input in inputs:
                output = self.from_cache(input)
                if not output:
                    output = func(self, input, callback=callback, *args, **kw)
                    self.add_cache(input, output)
                    
                result = result + ([output] if type(output) == Output else output)
                
        print("->", result)
                    

        self.output = result        
        if callback:
            callback(self)
            
        return result
    return runner_wrapper


## Other methodes
def same_types(a, b):
    if len(a.keys()) != len(b.keys()):
        return False
    for key, value in a.items():
        if isinstance(value, dict):
            return same_types(value, b[key])
        elif key not in b or value != b[key]:
            return False
    return True