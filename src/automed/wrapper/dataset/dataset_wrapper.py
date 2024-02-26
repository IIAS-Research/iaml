from automed.step import Step
from ...wrapper import StepWrapper


class DatasetWrapper(StepWrapper):
    def __init__(self, step: Step):
        super().__init__(step)

    def configure_one(self, config_id, key, value):
        if key in self.configurations:
            self.configure_one(config_id, key, value)
        else:
            self.step.configure_one(config_id, key, value)
