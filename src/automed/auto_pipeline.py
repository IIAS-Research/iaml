"""
Based on Scikit-learn Pipeline but for AutoMed Pipelines !
Transform, resample and then predict from Candidate instance 
"""
import pickle
import json
from hashlib import md5
from typing import TYPE_CHECKING
import pandas as pd
from sklearn.pipeline import Pipeline
from .dataset import Dataset

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step

class AutoPipeline(Pipeline):
    """
    Based on Scikit-learn Pipeline but for AutoMed Pipelines !
    Transform, resample and then predict from Candidate instance 
    """
    
    def __init__(
        self,
        steps: list[tuple[str, object]] = None
    ) -> None:
        """
        Args:
            steps (list[tuple[str, object]], optional): Ordered list of Automed.Steps.
                                                        Defaults to None.
        """
        if steps is None:
            steps = []
            
        self.transformers:list[tuple[str, object]] = []
        self.resamplers:list[tuple[str, object]] = []
        self.predictor:tuple[str, object] = None
        
        super().__init__(steps.copy())
        
    @property
    def steps(self):
        """
        Steps used to fit pipeline (same as steps property but with resamplers)

        Returns:
            list[tuple[str, object]]: list of steps
        """
        return [item for item in [*self.transformers, self.predictor] if item is not None]
    
    @property
    def training_steps(self) -> list[tuple[str, object]]:
        """
        Steps used to fit pipeline (same as steps property but with resamplers)

        Returns:
            list[tuple[str, object]]: list of steps
        """
        return [item for item in [*self.transformers, *self.resamplers, self.predictor] \
            if item is not None]
    
    @steps.setter
    def steps(self, values:list[tuple[str, object]]) -> list[tuple[str, object]]:
        for value in values:
            self.__add_step(value)
            
        return self.steps
            
    def __add_step(self, step:tuple[str, object]) -> None:
        _, instance = step
        if hasattr(instance, 'transform') and callable(instance.transform):
            self.transformers.append(step)
        elif hasattr(instance, 'predict') and callable(instance.predict):
            self.predictor = step
        elif hasattr(instance, 'resample') and callable(instance.resample):
            self.resamplers.append(step)
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, 
            only_predictor:bool=False, **kwargs) -> 'AutoPipeline':
        """
        Fit Pipeline on new data (or with new parameters)
        
        Args:
            X (pd.DataFrame): Candidate features
            y (pd.DataFrame): label to predict
        """
        if only_predictor:
            dataset = Dataset(X, y)
            self.predictor[1].fit(dataset)
        else:
            self.fit_transform(X, y)
        
        return self
    
    def fit_transform(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs) -> 'AutoPipeline':
        """
        Fit Pipeline and transform data 
        
        Args:
            X (pd.DataFrame): Candidate features
            y (pd.DataFrame): label to predict
        """
        dataset = Dataset(X, y)
        for _, step in [*self.transformers, *self.resamplers]:
            if 'Step' in map(lambda s: s.__name__, step.__class__.__mro__):
                step.fit(dataset)
            else:
                step.fit(dataset.X, dataset.y, **kwargs)
                
            if hasattr(step, 'transform'):
                dataset = Dataset(step.transform(dataset.X), y)
            elif hasattr(step, 'resample'):
                dataset = Dataset(*step.resample(dataset.X, dataset.y))
        
        return dataset.X, dataset.y
        
    @property
    def explanations(self):
        """
        Get explanations from all pipeline steps

        Returns:
            list[str]: List of markdown explanations
        """
        return [step.explain() for _, step in self.training_steps]
    
    @property
    def model(self) -> 'Step':
        """Shortcut to get the prediction model of AutoPipeline 

        Returns:
            Step: Prediction model of the pipeline (or None)
        """
        return self.predictor

    def add_transform(self, instance:'Step') -> None:
        """
        Add transform Step to the Pipeline

        Args:
            instance (Step): Step to add (must implement transform)
        """
        if instance and hasattr(instance, 'transform'):
            self.transformers.append((str(instance), instance))
        else:
            raise ValueError("Step must implement transform method")
        
    def add_resample(self, instance:'Step') -> None:
        """
        Add resample Step to the Pipeline

        Args:
            instance (Step): Step to add (must implement resample)
        """
        if instance and hasattr(instance, 'resample'):
            self.resamplers.append((str(instance), instance))
        else:
            raise ValueError("Step must implement resample method")
                
    def set_model(self, instance) -> None:
        """
        Set the predict model (Step) of the Pipeline

        Args:
            instance (Step): Model Step to add (must implement predict)
        """
        self.predictor = (str(instance), instance)

    def copy(self) -> 'AutoPipeline':
        """
        Return a copied AutoPipeline

        Returns:
            AutoPipeline: Copied AutoPipeline instance
        """
        return AutoPipeline(self.training_steps)
    

    def pickle(self) -> bytes:
        """
        Serialize AutoPipeline to bytes.
        Can be save into a file and reload with pickle.

        Returns:
            bytes: Serialized AutoPipeline
        """
        return pickle.dumps(self)
    
    @property
    def have_model(self) -> bool:
        """
        Does the AutoPipeline have a model set ?

        Returns:
            bool: True a model have been set
        """
        return bool(self.predictor)
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """Apply transformers without predict

        Args:
            X (pd.DataFrame): candidate data

        Returns:
            pd.DataFrame: transformed DF
        """
        for _, step in self.transformers:
            X = step.transform(X)
            
        return X
    
    def predict(self, X:pd.DataFrame, model_only:bool = False, **kwargs) -> list:
        """
        Run all the steps to predict labels from candidate data

        Args:
            X (pd.DataFrame): Features used as candidate of the pipeline
            model_only (bool, optional): True to execute only the model with already
                                        transformed data. Defaults to False.

        Raises:
            ValueError: Model must have been set before call predict

        Returns:
            list: Predicted values
        """
        if not self.have_model:
            raise ValueError("Model need to be set before predict")
        
        if not model_only:
            return super().predict(X, **kwargs)
        
        return self.model[1].predict(X)
    
    def __eq__(self, other: 'AutoPipeline') -> bool:
        return self.fingerprint() == other.fingerprint()
    
    def fingerprint(self) -> str:
        """
        Return a md5 hash that can by use to compare Pipelines 

        Returns:
            str: md5 sting
        """
        to_hash = "\n".join([str(step.__class__) + " = " \
            + json.dumps(step.configuration, sort_keys=True) \
                for _, step in self.training_steps])
        
        return md5(to_hash.encode()).hexdigest()
            
