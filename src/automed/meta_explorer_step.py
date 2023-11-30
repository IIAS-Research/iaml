from .metastep import MetaStep
from .output import Input
from .step import Step, isStep, runner
from .thread_with_return_value import *

#
# Inherit from MetaStep but will execute all steps at the same time. 
# The goal here is to explore many answer to a question. For example -> Try all Learning models
#
@isStep('meta')
class MetaExplorerStep(MetaStep):
    name = "MetaExplorerStep"
    def __init__(self, destroyer=None, *args, **kw):
        self.output = []
        
        if destroyer:
            destroyer.set_root(self)
            self.destroyers.append(destroyer)
        
    # Run a single Step
    # Will be executed by Theads
    def single_run(self, step, input, callback=None):
        self.output = self.output + (step.run(input, callback=callback))
        
    def json_pipeline(self):
        json = Step.json_pipeline(self)
        json['children'] = list(map(lambda step: step.json_pipeline(), self.steps))
        return json
        
    
    # Explore all steps
    @runner
    def run(self, input: Input, callback=None):
        output = []
        threads = []
        
        # Create one thread by Step
        for step in self.steps:
            threads.append(ThreadWithReturnValue(target=step.run, args=(input.to_input(model=input.model.copy()), callback)))
            
        # Run all threads
        for thread in threads:
            thread.start()
            
        # Wait end of all threads
        for thread in threads:
            output = output + thread.join()
        
        return output
            
            
        