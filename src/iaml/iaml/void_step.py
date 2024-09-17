"""
[STEP] Void step -> Just a step that do nothing and can be mutated to siblings
"""
from .step import Step
from .decorators.all import is_step


@is_step()
class VoidStep(Step):
    """
    [STEP] Void step -> Just a step that do nothing and can be mutated to siblings
    """
    name = "VoidStep"
    def __init__(self, *args, step_to_mimic:Step=None, **kwargs):  # pylint: disable=unused-argument
        if step_to_mimic:
            self.tags = step_to_mimic.tags
            self.step_to_mimic = step_to_mimic
            self.is_interchangeable = True
            self.optimizable = True
            
            if hasattr(step_to_mimic, 'predict') and callable(step_to_mimic.predict):
                def predict(X):
                    return X
                self.predict = predict
            elif hasattr(step_to_mimic, 'transform') and callable(step_to_mimic.transform):
                def transform(dataset):
                    return dataset
                self.transform = transform
            elif hasattr(step_to_mimic, 'resample') and callable(step_to_mimic.resample):
                def resample(X,y):
                    return X, y
                self.resample = resample
                
    @classmethod
    def from_pipeline(cls, pipeline:dict, *args, **kwargs) -> Step:
        """
        Load any VoidStep from json pipeline

        Args:
            pipeline (dict): Pipeline in a JSON format

        Raises:
            TypeError: invalid pipeline: VoidStep must have a Step to mimic

        Returns:
            Step: Step created from Json pipeline
        """
        if not 'step_to_mimic' in pipeline:
            raise TypeError('invalid pipeline: VoidStep must have a Step to mimic')
        
        to_mimic = Step.from_pipeline(pipeline['step_to_mimic'])
        
        step = super().from_pipeline(pipeline, step_to_mimic=to_mimic)
        
        return step
    
    
    def json_pipeline(self) -> dict:
        """
        Create JSON pipeline

        Returns:
            dict: Pipeline in JSON format
        """
        return {
            **Step.json_pipeline(self),
            'step_to_mimic': self.step_to_mimic.json_pipeline()
        }

    
