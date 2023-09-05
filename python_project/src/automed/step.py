class Step:
    # Available steps
    available_steps = {}
    
    output_dataset = None
    configuration = {}
    
    def __init__(self):
        self.default_configuration()
    
    
    ################
    # Configurable #
    ################
    #
    # A actionable step can be configured by the user.
    # For example, we can configure learning rate of a machine learning step.
    #  
    # Each parameters have a name, a description and a default value. Default value can be fixed or computed based on dataset
    
    def configure_one(self, key, value):
        if key in self.configuration.keys():
            self.configuration[key]['value'] = value
        else:
            raise(f"Configurable Key '#{key}' does not exist.")
        
    def configure(self, dict):
        for key, value in dict.items():
            self.configure_one(key, value)
    
    def resume_configuration(self):
        return {k: v['value'] for k, v in self.configuration.items()}
    
    # Alias for resume_configuration()
    def get_config(self):
        return self.resume_configuration()
    
    def default_values(self, dataset):
        return {k: v['default'] for k, v in self.configuration.items()}
    
    
    def default_configuration(self):
        for key, elem in self.configuration.items():
            self.configuration[key]['value'] = self.configuration[key]['default']
    
    
    ############
    # Priorize #
    ############
    #
    # Base on the dataset (or not) priorize usefulness of this actionable.
    # Result is a value between 0 and 1. 0 stand for not useful
    #
    def priorize(self, dataset=None):
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
    def runner_wrapper(self, dataset, *args, **kw):
        result = func(self, dataset, *args, **kw)
        self.output_dataset = result
        return result
    return runner_wrapper


    