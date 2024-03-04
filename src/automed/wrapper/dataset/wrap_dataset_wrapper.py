"""
[WRAPPER] Dataset Wrapper 
"""
from automed.step import Step
from ...step_wrapper import StepWrapper

class WrapDatasetWrapper(StepWrapper):
    """
    [WRAPPER] Dataset Wrapper 
    Use to pass configuration through wrapped steps
    """
    def __init__(self, step:Step): # pylint: disable=unused-argument
        self.learning_configuration:dict = self.step.configurations[0]

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
