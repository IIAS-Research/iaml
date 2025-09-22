"""Based on Scikit-learn Pipeline but for IAML Pipelines !
Transform, resample and then predict from Candidate instance 
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import pickle

from copy import deepcopy
from hashlib import md5

import numpy as np
import shap
import pandas as pd

from sklearn.pipeline import Pipeline

from .dataset import Dataset
from .void_step import VoidStep
from .explanation import Explanation
from .cache import Cache
from .reference import Reference

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step

class IAMLPipeline(Pipeline):
    """Based on Scikit-learn Pipeline but for IAML Pipelines !
    Transform, resample and then predict from Candidate instance.
    
    :param list[tuple[str, Step]], optional steps: Ordered list of IAML.Steps. Defaults to None.
    :param pd.DataFrame, optional original_dataset: Untransformed dataset to use as a masker 
                for the SHAP explainer which will be used to explain the model later on. 
                Defaults to None. If not provided, the prediction dataset will be used as the 
                masker, which may impact the accuracy of the explanations.
    :param str, optional estimator_type: Type of estimator. Must be one of 'classifier', 
        'survival', 'regressor'.
    """

    def __init__(
        self,
        steps: list[tuple[str, Step]] = None,
        original_dataset: pd.DataFrame = None,
        estimator_type: str = None) -> None:
        if steps is None:
            steps = []

        self.original_dataset: pd.DataFrame = original_dataset
        """Original dataset used for this pipeline"""

        self.transformers: list[tuple[str, object]] = []
        """List of transformers that'll be used in this pipeline."""

        self.resamplers: list[tuple[str, object]] = []
        """List of resamplers that'll be used in this pipeline."""

        self.predictor: tuple[str, object] = None
        """Predictor that'll be used in this pipeline"""

        self.metrics: list[Metric] = []
        """List of metrics that'll be computed in this pipeline"""

        if estimator_type not in ['classifier', 'regressor', 'survival']:
            raise ValueError(f"Estimator type ({estimator_type}) must be classifier, \
                survival or regressor")
        self.__estimator_type: str = estimator_type
        """Estimator type"""

        super().__init__(steps) # split steps into transformers, resamplers and predictor

    @property
    def _estimator_type(self) -> str:
        """Needed because used by Scikit-learn metalearner
        
        :return: estimator type.
        """
        return self.__estimator_type

    @property
    def estimator_type(self) -> str:
        """Needed because used by Scikit-learn metalearner (yes also without "_" ...)
        
        :return: estimator type.
        """
        return self.__estimator_type

    @property
    def steps(self) -> list[tuple[str, object]]:
        """Get all pipeline steps.
        
        :return: list of steps
        """
        return [item for item in [*self.transformers, self.predictor] if item is not None]

    @property
    def training_steps(self) -> list[tuple[str, object]]:
        """Steps used to fit pipeline (same as steps property but with resamplers)

        :return: list of steps
        """
        return [item for item in [*self.resamplers, *self.transformers, self.predictor] \
            if item is not None]

    @steps.setter
    def steps(self, values: list[tuple[str, object]]) -> list[tuple[str, object]]:
        """Set pipeline steps
        
        :param list[tuple[str, object]] values: List of steps to add.
        :return: List of new steps.
        """
        self.transformers = []
        self.resamplers = []
        self.predictor = None

        for value in values:
            self.__add_step(value)

        return self.steps

    def __add_step(self, step: tuple[str, object]) -> None:
        """Add a step to the pipeline steps
        
        :param tuple[str,object] step: The step to add.
        """
        _, instance = step
        if hasattr(instance, 'predict') and callable(instance.predict):
            self.predictor = step
        elif hasattr(instance, 'transform') and callable(instance.transform):
            self.transformers.append(step)
        elif hasattr(instance, 'resample') and callable(instance.resample):
            self.resamplers.append(step)

    def replace_step(self, old: 'Step', new: 'Step') -> bool:
        """Replace a step in the pipeline by another (by id)

        :param Step old: The step to replace.
        :param Step new: The new step.
        :return: Was replaced ?
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

    def remove_step(self, to_remove: 'Step') -> bool:
        """Remove a step from the pipeline (by object id)

        :param Step to_remove: Step to remove.
        :return: Step was removed ?
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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        only_predictor: bool = False,
        groups_columns: list[str] = None,
        metrics: list[Metric] = None,
        **kwargs: dict) -> 'IAMLPipeline':
        """Fit Pipeline on new data (or with new parameters)
        
        :param pd.DataFrame X: Candidate features.
        :param pd.DataFrame, optional y: label to predict. Default to None.
        :param bool, optional only_predictor: Run predictions only. Default to False.
        :param list[str], optional groups_columns: Columns name to use in splitting. 
            Default to None.
        :param list[Metric], optional metrics: List of Metrics to compute. Default to None.
        :param dict, optional \\**kwargs: Additional parameters.
        :return: Fitted IAMLPipeline.
        """
        self.metrics = metrics

        if groups_columns is None:
            groups_columns = []

        if not only_predictor:
            X, y = self.fit_transform(X, y, groups_columns=groups_columns, **kwargs)
            # Reset groups_columns as returned X is aldready pruned from groups columns
            # This way we avoid caching KeyError in dataset init
            groups_columns = []
        dataset = Dataset(X, y, groups_columns=groups_columns)

        if self.predictor[1].suitable(dataset):
            self.predictor[1].fit(dataset, **kwargs)
        else:
            self.predictor = None

        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        groups_columns: list[str] = None,
        **kwargs: dict) -> 'IAMLPipeline':
        """Fit Pipeline and transform data 
        
        :param pd.DataFrame X: Candidate features.
        :param pd.DataFrame, optional y: Label to predict. Default to None.
        :param list[str], optional groups_columns: Columns name to use in splitting. 
            Default to None.
        :param dict, optional \\**kwargs: Additional parameters.
        :return: Fitted IAMLPipeline.
        """
        if groups_columns is None:
            groups_columns = []

        dataset = Dataset(X, y, groups_columns=groups_columns)

        for _, step in [*self.resamplers, *self.transformers]:
            # FIT
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

            # APPLY TRANSFORM / RESAMPLE
            dataset_from_cache = Cache().from_cache(f"apply_{step.fingerprint()}", dataset.X)
            if dataset_from_cache:
                dataset = dataset_from_cache
            else:
                prev_X = dataset.X.copy()
                if hasattr(step, 'transform'):
                    dataset = Dataset(step.transform(dataset.X), dataset.y)
                elif hasattr(step, 'resample'):
                    dataset = Dataset(*step.resample(dataset.X, dataset.y))
                Cache().add_to_cache(f"apply_{step.fingerprint()}", prev_X, dataset)

        return dataset.X, dataset.y

    @property
    def explanations(self) -> list[str]:
        """Get explanations from all pipeline steps

        :return: List of markdown explanations
        """
        return [ e for _, step in self.training_steps if (e := step.explain()) is not None ]

    @property
    def model(self) -> Step:
        """Shortcut to get the prediction model of IAMLPipeline 

        :return: Prediction model of the pipeline (or None)
        """
        return self.predictor

    def add_transform(self, instance: Step) -> None:
        """Add transform Step to the Pipeline

        :param Step instance: Step to add (must implement transform).
        :raise ValueError: Step must implement transform method.
        """
        if instance and hasattr(instance, 'transform'):
            self.transformers.append((str(instance), instance))
        else:
            raise ValueError("Step must implement transform method")

    def add_resample(self, instance: Step) -> None:
        """Add resample Step to the Pipeline

        :param Step instance: Step to add (must implement resample).
        :raise ValueError: Step must implement resample method.
        """
        if instance and hasattr(instance, 'resample'):
            self.resamplers.append((str(instance), instance))
        else:
            raise ValueError("Step must implement resample method")

    def set_model(self, instance: Step) -> None:
        """
        Set the predict model (Step) of the Pipeline

        :param Step instance: Step to add (must implement predict).
        :raise ValueError: Step must implement predict method.
        """
        self.predictor = (str(instance), instance)

    def copy(self) -> IAMLPipeline:
        """Return a copied IAMLPipeline

        :return: Copied IAMLPipeline instance.
        """
        return deepcopy(self)

    def pickle(self) -> bytes:
        """Serialize IAMLPipeline to bytes.
        Can be save into a file and reload with pickle.

        :return: Serialized IAMLPipeline.
        """
        return pickle.dumps(self)

    @property
    def have_model(self) -> bool:
        """Does the IAMLPipeline have a model set?

        :return: True if a model has been set.
        """
        return bool(self.predictor)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: # pylint: disable=arguments-differ
        """Apply transformers without predict

        :param pd.DataFrame X: candidate data.
        :return: Transformed DF.
        """
        for _, step in self.transformers:
            X = step.transform(X)

        return X

    def predict(self, X: pd.DataFrame, model_only: bool = False, **kwargs: dict) -> list:
        """Run all the steps to predict labels from candidate data

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :param bool, optional model_only: True to execute only the model with already transformed 
            data. Defaults to False.
        :param dict, optional \\**kwargs: Additional parameters.
        :raise ValueError: Model must have been set before call predict.
        :return: Predicted values
        """
        if not self.have_model:
            raise ValueError("Model need to be set before predict")

        if not model_only:
            return super().predict(X, **kwargs)

        return self.predictor[1].predict(X)

    def predict_survival_function(
        self,
        X: pd.DataFrame,
        model_only: bool = False,
        **kwargs) -> list:
        """Run all the steps to predict survival function  from candidate data

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :param bool, optional model_only: True to execute only the model with already transformed 
            data. Defaults to False.
        :param dict, optional \\**kwargs: Additional parameters.
        :raise ValueError: Model must have been set before call predict.
        :return: Predicted values
        """
        if not self.have_model:
            raise ValueError("Model need to be set before predict")

        if not model_only:
            X = self.transform(X, **kwargs)

        return self.predictor[1].predict_survival_function(X)

    def predict_proba(self, X: pd.DataFrame, model_only: bool = False, **kwargs) -> list:
        """Run all the steps to predict labels from candidate data

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :param bool, optional model_only: True to execute only the model with already transformed 
            data. Defaults to False.
        :param dict, optional \\**kwargs: Additional parameters.
        :raise ValueError: Model must have been set before call predict.
        :return: Predicted values
        """
        if not self.have_model:
            raise ValueError("Model need to be set before predict")

        if not model_only:
            return super().predict_proba(X, **kwargs)

        return self.predictor[1].predict_proba(X)

    def __getattribute__(self, attr: str) -> bool:
        """Overload getattr to allow accurate hasattr on predict_proba

        :param str attr: Attribute to test.
        :raise AttributeError: predict_proba not implemented in this model.
        :return: Does attribute is implemented.
        """
        if attr == 'predict_proba' \
            and not( \
                self.have_model and hasattr(self.predictor[1], 'predict_proba') \
            ):
            raise AttributeError("predict_proba not implemented in this model")

        return super().__getattribute__(attr)

    @property
    def optimizable_step(self) -> list['Step']:
        """Return a list of optimizable step.
        
        :return: List of optimizable step
        """
        return [step for _, step in self.training_steps if step.optimizable]

    def __eq__(self, other: 'IAMLPipeline') -> bool:
        """Compare two pipelines

        :param IAMLPipeline other: The pipeline to compare.
        :return: Equal or not ?
        """
        if isinstance(other, IAMLPipeline):
            return self.fingerprint() == other.fingerprint()
        return NotImplemented

    @property
    def name(self) -> str:
        """Return pipeline formatted name
        
        :return: formatted name
        """
        return ' '.join(x.title() for x in str(self.model[0]).split('_'))

    def explain_model(self, X: pd.DataFrame, nsamples: int = 20) -> Explanation:
        """Explains the model by computing SHAP values on the fitted model.
        Uses the train set as the masker, and the provided set as
        prediction.

        :param pd.DataFrame X: Prediction set to compute SHAP values for.
        :param int, optional nsamples: Number of samples to pick from the masker to pick feature 
            data from for each row in the provided prediction dataset. More samples means more 
            accurate SHAP values and longer computing times. Defaults to 20.
        :raise RuntimeError: There is no model to explain.
        :return: Model explanation, with an overview of the most important features, and graphs.
        """
        if not self.have_model:
            raise RuntimeError('There is no model to explain.')

        def p(pred_data):
            df = pd.DataFrame(pred_data, columns=X.columns)

            if hasattr(self, 'predict_proba'):
                return self.predict_proba(df)[:, 1]

            # when the regressor does not implement predict_proba
            return self.predict(df)

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

        return Explanation(shap_explanation)

    # Implement scikit-learn estimator's methods
    def __sklearn_is_fitted__(self):
        return self.have_model

    def __sklearn_clone__(self):
        return deepcopy(self)

    def target_type_(self) -> str:
        """Mimic Scikit-learn API
        Return models target type
        """
        return self.original_dataset.type_of_target

    # Fingerprint (used by cache)
    def fingerprint(self) -> str:
        """Return a md5 hash that can by use to compare Pipelines 

        :return: md5 sting
        """
        to_hash = "\n".join([step.fingerprint() for _, step in self.training_steps])

        return md5(to_hash.encode()).hexdigest()

    def bibliography(self, structured: bool) -> str | list[dict]:
        """Return a string listing all step's references or a structured list of dict.
        
        :param bool structured: JSON structured bibliography or not.
        :return str | list[dict]: Bibliography.
        """
        references = [reference for step in self.steps
            for reference in step[1].references] \
            + [reference for metric in self.metrics for reference in metric.get_refs()] \
            + [Reference({
                'year': 2017,
                'name': 'A Unified Approach to Interpreting Model Predictions',
                'authors': [
                    'Scott M. Lundberg', 'Su-In Lee'
                ],
                'doi': 'https://doi.org/10.48550/arXiv.1705.07874',
                'publisher': 'arXiv preprint arXiv:1705.07874'
            }, 'Shap')]

        return Reference.bibliography(references, structured)
