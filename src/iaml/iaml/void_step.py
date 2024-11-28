"""
[STEP] Void step -> Just a step that do nothing and can be mutated to siblings
"""
from typing import Tuple, List, Dict
import pandas as pd
from .step import Step
from .decorators.all import is_step
from .dataset import Dataset


def predict(X: pd.DataFrame) -> pd.DataFrame:
    """
    VoidStep : Do nothing
    
    :param pd.DataFrame X: The dataframe we predict on
    
    :return: The dataframe returned
    """
    return X


def transform(dataset: Dataset) -> Dataset:
    """
    VoidStep : Do nothing
    
    :param Dataset dataset: The Dataset object we transform
    
    :return: The transformed Dataset
    """
    return dataset


def resample(X: pd.DataFrame, y: List) -> Tuple[pd.DataFrame, List]:
    """
    VoidStep : Do nothing
    
    :param pd.DataFrame X: The dataframe to resample
    :param List y: The dataframe target to resample
    
    :return: The resampled X and y
    """
    return X, y


@is_step()
class VoidStep(Step):
    """
    [STEP] Void step -> Just a step that do nothing and can be mutated to siblings
    
    :param Tuple, optional args: Additional parameters
    :param Step, optional step_to_mimic: The step to mimic, if provided
    :param Dict, optional kwargs: Additional parameters

    """
    name = "VoidStep"
    def __init__(self, *args, step_to_mimic: Step = None, **kwargs) -> None:  # pylint: disable=unused-argument
        """VoidStep : Do nothing
        """
        if step_to_mimic:
            self.tags = step_to_mimic.tags
            self.step_to_mimic = step_to_mimic
            self.is_interchangeable = True
            self.optimizable = True
            
            if hasattr(step_to_mimic, 'predict') and callable(step_to_mimic.predict):
                self.predict = predict
            elif hasattr(step_to_mimic, 'transform') and callable(step_to_mimic.transform):
                self.transform = transform
            elif hasattr(step_to_mimic, 'resample') and callable(step_to_mimic.resample):
                self.resample = resample
                
    @classmethod
    def from_pipeline(cls, pipeline: Dict, *args, **kwargs) -> Step:
        """Load any VoidStep from json pipeline

        :param Dict pipeline: Pipeline in a JSON format

        :raise TypeError: Invalid pipeline: VoidStep must have a Step to mimic

        :return: Step created from Json pipeline
        """
        if not 'step_to_mimic' in pipeline:
            raise TypeError('invalid pipeline: VoidStep must have a Step to mimic')
        
        to_mimic = Step.from_pipeline(pipeline['step_to_mimic'])
        
        step = super().from_pipeline(pipeline, step_to_mimic=to_mimic)
        
        return step
    
    
    def json_pipeline(self) -> Dict:
        """Create JSON pipeline

        :return: Pipeline in JSON format
        """
        return {
            **Step.json_pipeline(self),
            'step_to_mimic': self.step_to_mimic.json_pipeline()
        }
