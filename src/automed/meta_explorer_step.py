"""
[METASTEP] Explore all sub steps in Thread and return one output by Sub Step
"""
from .metastep import MetaStep
from .output import Input, Output
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
    [METASTEP] Explore all sub steps in Thread and return one output by Sub Step
    """
    name = "MetaExplorerStep"
    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.output = []
    
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
    def run(self, input_data:Input, callback:callable=None) -> Output:
        """
        Run all children Step in threads 

        Args:
            input_data (Input): Input data
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            Output: Result output
        """
        output = []
        workers: list[WorkerFuture] = []

        for step in self.steps:
            future = WorkerManager().submit(step, step.run, input_data.to_input(), callback)
            workers.append(future)
            
        # Wait end of all threads
        for worker in workers:
            output = output + worker.result()
        
        return output
