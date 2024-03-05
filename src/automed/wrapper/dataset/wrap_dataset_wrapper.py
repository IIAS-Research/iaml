"""
[WRAPPER] Dataset Wrapper 
"""
from typing import TYPE_CHECKING
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
        self.learning_configuration:dict = self.step.configurations[0]

    def run(self, input_data: 'Input', callback: callable = None) -> None:
        metrics = { m: input_data.computed_metrics[str(m)] for m in input_data.metrics }
        input_data.pipeline.add_explanation(self.step, None, metrics)

        return input_data

    def configure_one(self, config_id:int, key: str, value:any) -> None:
        """
        Configure this step or wrapped step

        Args:
            config_id (int): Index of configuration
            key (str): Key of parameter
            value (any): Value to set
        """
        if key not in self.configurations[0]:
            self.step.configure_one(config_id, key, value)
        else:
            super().configure_one(config_id, key, value)
