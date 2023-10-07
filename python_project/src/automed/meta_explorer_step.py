from .step import Step, isStep, runner
from .metastep import MetaStep

from threading import Thread

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
            threads.append(ThreadWithReturnValue(target=step.run, args=(input, callback)))
            
        # Run all threads
        for thread in threads:
            thread.start()
            
        # Wait end of all threads
        for thread in threads:
            output = self.output + thread.join()
            
        # print("=>>", list(map(lambda x: x.computed, output)))
        
        return output
            
            
            
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return