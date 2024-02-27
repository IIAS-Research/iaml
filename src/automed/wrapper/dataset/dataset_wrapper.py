from automed.step import Step
from ...wrapper import StepWrapper


class DatasetWrapper(StepWrapper):
    def __init__(self, step: Step):
        super().__init__(step)

        self.configurations = [{ f'_{k}': v for k, v in step.configurations[0].items() } | self.configurations[0]]

    def configure_one(self, config_id, key: str, value):
        if key.startswith('_'):
            self.step.configure_one(config_id, key[1:], value)

        super().configure_one(config_id, key, value)
