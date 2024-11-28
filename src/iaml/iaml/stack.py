"""
Frozen version of Step created to be stacked in an Candidate
"""

from typing import Dict
from .step import Step
class Stack:
    """
    Frozen version of Step created to be stacked in an Candidate
    
    :param Step step_class: The step class used to create this stack
    :param Dict configuration: The configuration dictionnary
    :param int step_id: The id of the Step
    """
    def __init__(self, step_class: Step, configuration: Dict, step_id: int) -> None:
        """Initialize a stack
        """
        self.step_class:Step = step_class
        """The step used in this Stack"""

        self.configuration:Dict = configuration
        """The step configuration"""

        self.step_id:int = step_id
        """The step id"""
        
    def __str__(self) -> str:
        """Return the step name as Stack representation
        
        :return: The step name
        """
        return self.step_class.name
    
    def explain(self) -> str:
        """Return explanation string of Step
        
        :return: The explanation of the Step in string format
        """
        return self.step_class.explain(self.configuration)
