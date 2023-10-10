from .step import Step, isStep, runner
from .metastep import MetaStep
from .thread_with_return_value import *

@isStep('meta')
class MetaExplorerStep(MetaStep):
    def __init__(self, destroyer=None, *args, **kw):
        self.output = []
        
        if destroyer:
            destroyer.set_root(self)
            self.destroyers.append(destroyer)
        
    # Will be executed by Theads
    def single_run(self, step, input, callback=None):
        self.output = self.output + (step.run(input, callback=callback))
        
    
    # Explore all steps
    @runner
    def run(self, input, callback=None):
        output = []
        threads = []
        
        # Create one thread by Step
        for step in self.steps:
            threads.append(ThreadWithReturnValue(target=step.run, args=(input, callback)))
            
        # Run all threads
        for thread in threads:
            thread.start()
            
        # Wait end of all threads
        for thread in threads:
            output = output + thread.join()
        
        return output
            
            
        