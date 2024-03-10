"""
[WRAPPER] Dataset Wrapper 
"""
from typing import Any, TYPE_CHECKING
from multipledispatch import dispatch
from automed.step import Step
from ...step_wrapper import StepWrapper

if TYPE_CHECKING:
    from ...output import Input

class WrapDatasetWrapper(StepWrapper):
    """
    [WRAPPER] Dataset Wrapper 
    Use to pass configuration through wrapped steps
    """
    def __init__(self, step: Step):
        super().__init__(step)
        self.learning_configuration:dict = self.step.configuration

    def run(self, input_data: 'Input', callback: callable = None) -> None:
        # metrics = { m: input_data.computed_metrics[str(m)] for m in input_data.metrics }
        # input_data.pipeline.add_explanation(self.step, None, metrics)

        return input_data

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
