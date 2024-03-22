"""
[METASTEP] Explore all sub steps in Thread and return one candidate by Sub Step
"""
from copy import deepcopy
from .metastep import MetaStep
from .candidate import Candidate
from .step import Step
from .decorators.all import is_step, runner
from .worker_manager import WorkerFuture, WorkerManager

#
# Inherit from MetaStep but will execute all steps at the same time. 
# The goal here is to explore many answer to a question. For example -> Try all Learning models
#
@is_step('meta')
class MetaExplorerStep(MetaStep):
    """
    [METASTEP] Explore all sub steps in Thread and return one candidate by Sub Step
    """
    name = "MetaExplorerStep"
    def __init__(self, *args, also_explore_without:bool=False, **kwargs):  # pylint: disable=unused-argument
        self.candidate = []
        self.also_explore_without = also_explore_without
    
    def json_pipeline(self) -> dict:
        """
        Create a JSON format of the pipeline

        Returns:
            dict: Pipeline in JSON format
        """
        json = Step.json_pipeline(self)
        json['children'] = list(map(lambda step: step.json_pipeline(), self.steps))
        return json
        
    
    # Explore all steps
    @runner
    def run(self, candidate:Candidate, callback:callable=None) -> Candidate:
        """
        Run all children Step in threads 

        Args:
            candidate (Candidate): Candidate data
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            Candidate: Result candidate
        """
        output = []
        
        # Try a candidate without any of explored steps 
        if self.also_explore_without:
            output += [candidate.to_output()]
            
        workers: list[WorkerFuture] = []

        for step in self.steps:
            future = WorkerManager().submit(step, step.run, candidate.to_input(), callback)
            workers.append(future)
            
        # Wait end of all threads
        for worker in workers:
            output += worker.result()
        
        return output
