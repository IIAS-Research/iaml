"""
StepWrapper is a direct child of Step and will wrap and execute another step.
Wrap with StepWrapper is useless, use children classes
"""
from typing import Dict, List
from .step import Step
from .decorators.all import is_step, runner
from .candidate import Candidate
from .dataset import Dataset


@is_step('wrapper')
class StepWrapper(Step):
    """
    StepWrapper is a direct child of Step and will wrap and execute another step.
    Wrap with StepWrapper is useless, use children classes
    """
    def __init__(self, step: Step):
        """
        Initialize the step wrapper
        
        Parameters
        ----------
        step : Step
            The step to initialize
        """
        self.step:Step = step
        
    
    @classmethod
    def from_pipeline(cls, pipeline: Dict, *args, **kwargs) -> Step:
        """
        Load any kind of StepWrapper

        Parameters
        ----------
        pipeline : Dict
            JSON pipeline

        Raises
        ------
        TypeError
            invalid pipeline: StepWrapper must have exactly one child

        Returns
        -------
        Step
            Loaded step
        """
        if 'children' not in pipeline or len(pipeline['children']) != 1:
            raise TypeError('invalid pipeline: StepWrapper must have exactly one child')
        
        child = Step.from_pipeline(pipeline['children'][0])
        step = super().from_pipeline(pipeline, child)
        
        return step

    
    def configure_parents(self, *parents) -> None:
        """
        Back propagate steps to parents
        
        Parameters
        ----------
        parents : Tuple[Step]
            Tuple of parents to configure
        """
        self.step.configure_parents(*parents)
        super().configure_parents(*parents)
        

    def wrap(self, step:Step) -> None:
        """
        Set wrapped step

        Parameters
        ----------
        step : Step
            Step to wrap

        Raises
        ------
        ValueError
            Step must be an occurrence of step (or inherited classes)
        """
        if Step in step.__class__.__mro__:
            self.step = step
        else:
            raise ValueError("Step must be an occurrence of step (or inherited classes)")
        
    def suitable(self, dataset:Dataset) -> bool:
        """
        Is suitable if the wrapped step is

        Parameters
        ----------
        candidate : Candidate
            Candidate data

        Returns
        -------
        bool
            Suitable?
        """
        return self.step.suitable(dataset)
    
    def all_configurations(self) -> List[Dict]:
        """
        Get configuration of all children Steps

        Returns
        -------
        List[Dict]
            All children configurations
        """
        to_return = Step.all_configurations(self)
        to_return = to_return + self.step.all_configurations()
        
        return to_return
    
    
    def json_pipeline(self) -> Dict:
        """
        Create a JSON pipeline

        Returns
        -------
        Dict
            JSON pipeline
        """
        return {
            **Step.json_pipeline(self),
            'children': [self.step.json_pipeline()]
        }
    
    
    def all_steps(self) -> List[Step]:
        """
        Recursive function to get all steps in a pipeline

        Returns
        -------
        List[Step]
            All children Step
        """
        return [self.step, *self.step.all_steps()]
        

    @runner
    def run(self, candidate:Candidate) -> Candidate:
        """
        This wrapper is useless. Only run the step
        
        Parameters
        ----------
        candidate : Candidate
            The candidate the step will run on
        
        Returns
        -------
        Candidate
            The transformed candidate
        """
        return self.step.run(candidate) 
    
    def count_steps(self) -> int:
        """
        Returns a rough estimation of the total count of steps for a given
        pipeline.
        
        Returns
        -------
        int
            Number of steps for a given pipeline
        """
        return 1 + self.step.count_steps()
    
    def priorize(self, candidate: Candidate=None) -> float:
        """
        Try to priorize himself

        Parameters
        ----------
        candidate : Candidate
            The candidate we try to priorize

        Returns
        -------
        float
            continuous between 0 and 1
        """
        return self.step.priorize(candidate)
