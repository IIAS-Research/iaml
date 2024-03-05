"""
[WRAPPER] Dataset Wrapper 
"""
from typing import Any
from multipledispatch import dispatch
from automed.step import Step
from ...step_wrapper import StepWrapper

class WrapDatasetWrapper(StepWrapper):
    """
    [WRAPPER] Dataset Wrapper 
    Use to pass configuration through wrapped steps
    """
    def __init__(self, step:Step): # pylint: disable=unused-argument
        super().__init__(step)
        self.learning_configuration:dict = self.step.configuration

    @dispatch(str, object)
    def configure(self, key: str, value:Any) -> None:
        """
        Configure this step or wrapped step

        Args:
            key (str): Key of parameter
            value (any): Value to set
        """
        if key not in self.configuration:
            self.step.configure(key, value)
        else:
            super().configure(key, value)
