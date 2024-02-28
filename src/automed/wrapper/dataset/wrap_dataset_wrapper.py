from automed.step import Step
from .. import StepWrapper


class WrapDatasetWrapper(StepWrapper):
    def __init__(self, step: Step):
        super().__init__(step)

        self.learning_configuration = self.step.configurations[0]

    def configure_one(self, config_id, key: str, value):
        if key not in self.configurations[0]:
            return self.step.configure_one(config_id, key, value)
        
        super().configure_one(config_id, key, value)
