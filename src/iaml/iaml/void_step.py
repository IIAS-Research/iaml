"""
[STEP] Void step -> Just a step that do nothing and can be mutated to siblings
"""
from typing import Tuple, List
import pandas as pd
from .step import Step
from .decorators.all import is_step
from .dataset import Dataset


def predict(X:pd.DataFrame) -> pd.DataFrame:
    """
    VoidStep : Do nothing
    
    Parameters
    ----------
    X: pd.DataFrame
        The dataframe we predict on
    
    Returns
    -------
    pd.DataFrame
        The dataframe returned
    """
    return X


def transform(dataset:Dataset) -> Dataset:
    """
    VoidStep : Do nothing
    
    Parameters
    ----------
    dataset: Dataset
        The Dataset object we transform
    
    Returns
    -------
    Dataset
        The transformed Dataset
    """
    return dataset


def resample(X:pd.DataFrame,y:List) -> Tuple[pd.DataFrame, List]:
    """
    VoidStep : Do nothing
    
    
    Parameters
    ----------
    X : pd.DataFrame
        The dataframe to resample
    y : List
        The dataframe target to resample
    
    Returns
    -------
    Tuple[pd.DataFrame, List]
        The resampled X and y
    """
    return X, y


@is_step()
class VoidStep(Step):
    """
    [STEP] Void step -> Just a step that do nothing and can be mutated to siblings
    """
    name = "VoidStep"
    def __init__(self, *args, step_to_mimic:Step=None, **kwargs):  # pylint: disable=unused-argument
        """
        VoidStep : Do nothing
        
        
        Parameters
        ----------
        args : Tuple
            Optionnal parameters
        step_to_mimic : Step
            A specific step to mimic
        kwargs : Dict[str, Any]
            Optionnal dictionnary parameters
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
    def from_pipeline(cls, pipeline:dict, *args, **kwargs) -> Step:
        """
        Load any VoidStep from json pipeline

        Parameters
        ----------
        pipeline : Dict
            Pipeline in a JSON format

        Raises
        ------
        TypeError
            Invalid pipeline: VoidStep must have a Step to mimic

        Returns
        -------
        Step
            Step created from Json pipeline
        """
        if not 'step_to_mimic' in pipeline:
            raise TypeError('invalid pipeline: VoidStep must have a Step to mimic')
        
        to_mimic = Step.from_pipeline(pipeline['step_to_mimic'])
        
        step = super().from_pipeline(pipeline, step_to_mimic=to_mimic)
        
        return step
    
    
    def json_pipeline(self) -> dict:
        """
        Create JSON pipeline

        Returns
        -------
        Dict
            Pipeline in JSON format
        """
        return {
            **Step.json_pipeline(self),
            'step_to_mimic': self.step_to_mimic.json_pipeline()
        }
