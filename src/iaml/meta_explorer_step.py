"""[METASTEP] Explore all sub steps in Thread and return one candidate by Sub Step"""
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
    """[METASTEP] Explore all sub steps in Thread and return one candidate by Sub Step"""

    name: str = "MetaExplorerStep"
    _description: str = 'Execute all steps and keep the best result'
    _description_long: str = None

    def __init__(self, *args, also_explore_without: bool = False, **kwargs):  # pylint: disable=unused-argument
        self.candidate = []
        self.also_explore_without = also_explore_without

    def json_pipeline(self) -> dict:
        json = Step.json_pipeline(self)
        json['children'] = list(map(lambda step: step.json_pipeline(), self.steps))
        return json

    def add_step(self, step: Step) -> None:
        if Step in step.__class__.__mro__:
            step.is_interchangeable = True

            # Define enable only if False because True can generate strange behavior.
            # True is default value anyway
            if not self._enable:
                step.enable = False

            self.steps.append(self.configure_child(step))
        else:
            raise ValueError("step must be an occurrence of step (or inherited classes)")

    # Explore all steps
    @runner
    def run(self, candidate: Candidate) -> Candidate:
        output = []

        # Try a candidate without any of explored steps
        if self.also_explore_without:
            output += [candidate.to_output()]

        workers: list[WorkerFuture] = []

        for step in self.steps:
            future = WorkerManager().submit(step, step.run, candidate.to_input())
            workers.append(future)

        # Wait end of all threads
        for worker in workers:
            output += worker.result()

        return output
