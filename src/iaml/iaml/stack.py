"""
Frozen version of Step created to be stacked in an Candidate
"""

from typing import Dict
from .step import Step
class Stack:
    """
    Frozen version of Step created to be stacked in an Candidate
    """
    def __init__(self, step_class: Step, configuration: Dict, step_id: int) -> None:
        """
        Initialize a stack
        
        Parameters
        ----------
        step_class : Step
            The step class used to create this stack
        configuration : Dict
            The configuration dictionnary
        step_id : int
            The id of the Step
        """
        self.step_class = step_class
        self.configuration:Dict = configuration
        self.step_id:int = step_id
        
    def __str__(self) -> str:
        return self.step_class.name
    
    def explain(self) -> str:
        """
        Return explanation string of Step
        
        Returns
        -------
            The explanation of the Step in string format
        """
        return self.step_class.explain(self.configuration)
