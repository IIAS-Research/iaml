"""
[STEP] Void step -> Just a step that do nothing and can be mutated to siblings
"""
from .step import Step
from .decorators.all import is_step


def predict(X):
    return X


def transform(dataset):
    return dataset


def resample(X,y):
    return X, y


@is_step()
class VoidStep(Step):
    """
    [STEP] Void step -> Just a step that do nothing and can be mutated to siblings
    """
    name = "VoidStep"
    def __init__(self, *args, step_to_mimic:Step=None, **kwargs):  # pylint: disable=unused-argument
        if step_to_mimic:
            self.tags = step_to_mimic.tags
            self.is_interchangeable = True
            self.optimizable = True
            
            if hasattr(step_to_mimic, 'predict') and callable(step_to_mimic.predict):
                self.predict = predict
            elif hasattr(step_to_mimic, 'transform') and callable(step_to_mimic.transform):
                self.transform = transform
            elif hasattr(step_to_mimic, 'resample') and callable(step_to_mimic.resample):
                self.resample = resample
