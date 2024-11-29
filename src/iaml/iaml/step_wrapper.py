"""StepWrapper is a direct child of Step and will wrap and execute another step.
Wrap with StepWrapper is useless, use children classes
"""
from .step import Step
from .decorators.all import is_step, runner
from .candidate import Candidate
from .dataset import Dataset


@is_step('wrapper')
class StepWrapper(Step):
    """StepWrapper is a direct child of Step and will wrap and execute another step.
    Wrap with StepWrapper is useless, use children classes
    """
    def __init__(self, step: Step):
        self.step: Step = step
        """The step to wrap"""
        
    
    @classmethod
    def from_pipeline(cls, pipeline: dict, *args, **kwargs) -> Step:
        """
        Load any kind of StepWrapper. The step must have exactly one child

        :param dict pipeline: JSON pipeline.
        :raise TypeError: invalid pipeline: StepWrapper must have exactly one child.
        :return: Loaded step.
        """
        if 'children' not in pipeline or len(pipeline['children']) != 1:
            raise TypeError('invalid pipeline: StepWrapper must have exactly one child')
        
        child = Step.from_pipeline(pipeline['children'][0])
        step = super().from_pipeline(pipeline, child)
        
        return step

    
    def configure_parents(self, *parents) -> None:
        self.step.configure_parents(*parents)
        super().configure_parents(*parents)
        

    def wrap(self, step: Step) -> None:
        """Set wrapped step

        :param Step step: Step to wrap.
        :raise ValueError: Step must be an occurrence of step (or inherited classes).
        """
        if Step in step.__class__.__mro__:
            self.step = step
        else:
            raise ValueError("Step must be an occurrence of step (or inherited classes)")
        
    def suitable(self, dataset:Dataset) -> bool:
        return self.step.suitable(dataset)
    
    def all_configurations(self) -> list[dict]:
        to_return = Step.all_configurations(self)
        to_return = to_return + self.step.all_configurations()
        
        return to_return
    
    
    def json_pipeline(self) -> dict:
        return {
            **Step.json_pipeline(self),
            'children': [self.step.json_pipeline()]
        }
    
    
    def all_steps(self) -> list[Step]:
        return [self.step, *self.step.all_steps()]
        

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        return self.step.run(candidate) 
    
    def count_steps(self) -> int:
        return 1 + self.step.count_steps()
    
    def priorize(self, candidate: Candidate=None) -> float:
        return self.step.priorize(candidate)
