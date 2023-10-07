from .step import Step, isStep, runner
from .metastep import MetaStep

import threading

@isStep('meta')
class MetaExplorerStep(MetaStep):
    def __init__(self, *args, **kw):
        self.output = []
        
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
            threads.append(threading.Thread(target=self.single_run, args=(step, input, callback)))
            
        # Run all threads
        for thread in threads:
            thread.start()
            
        # Wait end of all threads
        for thread in threads:
            thread.join()
        
        return output
            