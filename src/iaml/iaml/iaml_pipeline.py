"""
Based on Scikit-learn Pipeline but for IAML Pipelines !
Transform, resample and then predict from Candidate instance 
"""
import pickle
from copy import deepcopy
from hashlib import md5
from typing import TYPE_CHECKING
import numpy as np
import shap
import pandas as pd
from sklearn.pipeline import Pipeline
from .dataset import Dataset
from .void_step import VoidStep
from .explanation import Explanation
from .cache import Cache

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step

class IAMLPipeline(Pipeline):
    """
    Based on Scikit-learn Pipeline but for IAML Pipelines !
    Transform, resample and then predict from Candidate instance 
    """
    
    def __init__(
        self,
        steps: list[tuple[str, 'Step']] = None,
        original_dataset: pd.DataFrame = None,
        estimator_type:str = None
    ) -> None:
        """
        Args:
            steps (list[tuple[str, Step]], optional): Ordered list of Automed.Steps.
                Defaults to None.
            explanations (list[Explanation], optional): List of Explanation objects.
                Defaults to None.
            original_dataset (pd.DataFrame, optional): Untransformed
                dataset to use as a masker for the SHAP explainer which
                will be used to explain the model later on. Defaults to
                None. If not provided, the prediction dataset will be
                used as the masker, which may impact the accuracy of
                the explanations.
        """
        if steps is None:
            steps = []
            
        self.original_dataset = original_dataset
        self.transformers:list[tuple[str, object]] = []
        self.resamplers:list[tuple[str, object]] = []
        self.predictor:tuple[str, object] = None
        
        if estimator_type not in ['classifier', 'regressor']:
            raise ValueError(f"Estimator type ({estimator_type}) must be classifier or regressor")
        self.__estimator_type = estimator_type
        
        super().__init__(steps) # split steps into transformers, resamplers and predictor
        
    @property
    def _estimator_type(self):
        """
        Needed because used by Scikit-learn metalearner
        """
        return self.__estimator_type
    
    @property
    def estimator_type(self):
        """
        Needed because used by Scikit-learn metalearner (yes also without "_" ...)
        """
        return self.__estimator_type
    
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
        self.transformers = []
        self.resamplers = []
        self.predictor = None
        for value in values:
            self.__add_step(value)
            
        return self.steps
            
    def __add_step(self, step:tuple[str, object]) -> None:
        _, instance = step
        if hasattr(instance, 'predict') and callable(instance.predict):
            self.predictor = step
        elif hasattr(instance, 'transform') and callable(instance.transform):
            self.transformers.append(step)
        elif hasattr(instance, 'resample') and callable(instance.resample):
            self.resamplers.append(step)
    
    def replace_step(self, old:'Step', new:'Step') -> bool:
        """Replace a step in the pipeline by another (by id)

        Args:
            old (Step): Old step to replace
            new (Step): New step

        Returns:
            bool: Was replaced ?
        """
        for idx, step in enumerate(self.transformers):
            if id(old) == id(step[1]):
                self.transformers[idx] = (new.name, new)
                return True
        for idx, step in enumerate(self.resamplers):
            if id(old) == id(step[1]):
                self.resamplers[idx] = (new.name, new)
                return True
        if id(old) == id(self.predictor[1]):
            self.predictor = (new.name, new)
            return True
        
        return False
        
    def remove_step(self, to_remove:'Step') -> bool:
        """Remove a step from the pipeline (by object id)

        Args:
            to_remove (Step): Step to remove

        Returns:
            bool: Step was removed ?
        """
        for idx, step in enumerate(self.transformers):
            if id(to_remove) == id(step[1]):
                del self.transformers[idx]
                return True
        for idx, step in enumerate(self.resamplers):
            if id(to_remove) == id(step[1]):
                del self.resamplers[idx]
                return True
        if id(to_remove) == id(self.predictor[1]):
            self.predictor = None
            return True
        
        return False
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, 
            only_predictor:bool=False, **kwargs) -> 'IAMLPipeline':
        """
        Fit Pipeline on new data (or with new parameters)
        
        Args:
            X (pd.DataFrame): Candidate features
            y (pd.DataFrame): label to predict
        """
        if not only_predictor:
            X, y = self.fit_transform(X, y, **kwargs)
            
        dataset = Dataset(X, y)
        
        if self.predictor[1].suitable(dataset):
            self.predictor[1].fit(dataset, **kwargs)
        else:
            self.predictor = None
        
        return self
    
    def fit_transform(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs) -> 'IAMLPipeline':
        """
        Fit Pipeline and transform data 
        
        Args:
            X (pd.DataFrame): Candidate features
            y (pd.DataFrame): label to predict
        """
        dataset = Dataset(X, y)
        
        for _, step in [*self.transformers, *self.resamplers]:
            if 'Step' in map(lambda s: s.__name__, step.__class__.__mro__):
                from_cache = Cache().from_cache(f"fit_{step.fingerprint()}", dataset.X)
                if from_cache:
                    self.replace_step(step, from_cache)
                else:
                    if step.suitable(dataset):
                        step.fit(dataset)
                        Cache().add_to_cache(f"fit_{step.fingerprint()}", dataset.X, step)
                    else:
                        if step.is_interchangeable:
                            old_step = step
                            step = VoidStep(step_to_mimic=step)
                            self.replace_step(old_step, step)
                        else:
                            self.remove_step(step)
                            continue
            else:
                step.fit(dataset.X, dataset.y, **kwargs)
                
            dataset_from_cache = Cache().from_cache(f"apply_{step.fingerprint()}", dataset.X)
            if dataset_from_cache:
                dataset = dataset_from_cache
            else:
                prev_X = dataset.X.copy()
                if hasattr(step, 'transform'):
                    dataset = Dataset(step.transform(dataset.X), y)
                elif hasattr(step, 'resample'):
                    dataset = Dataset(*step.resample(dataset.X, dataset.y))
                Cache().add_to_cache(f"apply_{step.fingerprint()}", prev_X, dataset)
                
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
        """Shortcut to get the prediction model of IAMLPipeline 

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

    def copy(self) -> 'IAMLPipeline':
        """
        Return a copied IAMLPipeline

        Returns:
            IAMLPipeline: Copied IAMLPipeline instance
        """
        return deepcopy(self)
    

    def pickle(self) -> bytes:
        """
        Serialize IAMLPipeline to bytes.
        Can be save into a file and reload with pickle.

        Returns:
            bytes: Serialized IAMLPipeline
        """
        return pickle.dumps(self)
    
    @property
    def have_model(self) -> bool:
        """
        Does the IAMLPipeline have a model set ?

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
        
        return self.predictor[1].predict(X)
    
    def predict_proba(self, X:pd.DataFrame, model_only:bool = False, **kwargs) -> list:
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
            return super().predict_proba(X, **kwargs)
        
        return self.predictor[1].predict_proba(X)

    @property
    def optimizable_step(self) -> list['Step']:
        """
        Returns:
            list[Step]: List of optimizable step
        """
        return [step for _, step in self.training_steps if step.optimizable]
    
    def __eq__(self, other: 'IAMLPipeline') -> bool:
        if isinstance(other, IAMLPipeline):
            return self.fingerprint() == other.fingerprint()
        return NotImplemented 
    
    def explain_model(self, X: 'pd.DataFrame', nsamples: int = 20):
        """
        Explains the model by computing SHAP values on the fitted model.
        Uses the train set as the masker, and the provided set as
        prediction.

        Args:
            X (DataFrame): Prediction set to compute SHAP values for.
            nsamples (int, optional): Number of samples to pick from
                the masker to pick feature data from for each row in
                the provided prediction dataset. More samples means
                more accurate SHAP values and longer computing times.
                Defaults to 20.

        Returns:
            Explanation: Model explanation, with an overview of the
                most important features, and graphs.
        """
        if not self.have_model:
            raise RuntimeError('There is no model to explain.')

        def p(pred_data):
            return self.predict_proba(pd.DataFrame(pred_data, columns=X.columns))[:, 1]

        mask_dataset = self.original_dataset if self.original_dataset is not None \
            and not self.original_dataset.empty else X

        explainer = shap.KernelExplainer(p, mask_dataset)
        shap_values = explainer.shap_values(X, nsamples=nsamples)

        shap_explanation = shap.Explanation(
            shap_values,
            base_values=np.tile(explainer.expected_value, (shap_values.shape[0], 1)),
            data=X.to_numpy(),
            feature_names=X.columns.to_list(),
            output_names=X.columns.to_list())

        return Explanation(self.model, None, None, shap_explanation)
    
    # Implement scikit-learn estimator's methods
        
    def __sklearn_is_fitted__(self):
        return self.have_model
    
    def __sklearn_clone__(self):
        return deepcopy(self)
    
    # Fingerprint (used by cache)
    def fingerprint(self) -> str:
        """
        Return a md5 hash that can by use to compare Pipelines 

        Returns:
            str: md5 sting
        """
        to_hash = "\n".join([step.fingerprint() for _, step in self.training_steps])
        
        return md5(to_hash.encode()).hexdigest()
            
