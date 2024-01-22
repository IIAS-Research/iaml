from .metastep import MetaStep
from .output import Input
from .step import Step, isStep, runner
from .worker_manager import WorkerFuture, WorkerManager

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
        workers: list[WorkerFuture] = []

        for step in self.steps:
            future = WorkerManager().submit(step, step.run, input.to_input(), callback)
            workers.append(future)
            
        # Wait end of all threads
        for worker in workers:
            output = output + worker.result()
        
        return output
