"""
Based on Scikit-learn Pipeline but for AutoMed Pipelines !
Transform, resample and then predict from Input instance 
"""
import pickle
from typing import TYPE_CHECKING
import numpy as np
import shap
from numpy import ndarray
import pandas as pd
from sklearn.pipeline import Pipeline
from .explanation import Explanation

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step

class AutoPipeline(Pipeline):
    """
    Based on Scikit-learn Pipeline but for AutoMed Pipelines !
    Transform, resample and then predict from Input instance 
    """
    
    def __init__(
        self,
        steps: list[tuple[str, 'Step']] = None,
        explanations: list[Explanation] = None,
        original_dataset: pd.DataFrame = None,
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

        if explanations is None:
            explanations = []

        self.steps:list[tuple[str, 'Step']] = steps.copy()
        self.explanations:list[Explanation] = explanations.copy()
        self.original_dataset = original_dataset
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs):
        """
        fit is not usable with AutoPipeline
        """
        raise NotImplementedError("fit() is not usable with AutoMed Pipeline. \
                                    You have to use AutoMed.run()")
    
    def fit_predict(self, X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs):
        """
        fit_predict is not usable with AutoPipeline
        """
        raise NotImplementedError("fit_predict is not usable with AutoMed Pipeline. \
                                    You have to use AutoMed.run()")
        
    def fit_transform(self, X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs):
        """
        fit_transform is not usable with AutoPipeline
        """
        raise NotImplementedError("fit_transform is not usable with AutoMed Pipeline. \
                                You have to use AutoMed.run()")
        
    @property
    def model(self) -> 'Step':
        """Shortcut to get the prediction model of AutoPipeline 

        Returns:
            Step: Prediction model of the pipeline (or None)
        """
        return self.steps[-1][1] if self.have_model else None
    
    @property
    def _estimator_type(self) -> str:
        return getattr('_estimator_type', self.model.model) if self.have_model else None

    def add_transform(self, instance:'Step') -> None:
        """
        Add transform Step to the Pipeline

        Args:
            instance (Step): Step to add (must implement transform)
        """
        if instance and hasattr(instance, 'transform'):
            if self.have_model: # Model must stay the last step
                self.steps.insert(-1, (str(instance), instance))
            else:
                self.steps.append((str(instance), instance))
        else:
            raise ValueError("Step must implement transform method")
                
    def set_model(self, instance) -> None:
        """
        Set the predict model (Step) of the Pipeline

        Args:
            instance (Step): Model Step to add (must implement predict)
        """
        
        if self.have_model: # Replace the previous model 
            self.steps.pop(-1)
            
        self.steps.append((str(instance), instance))

    def copy(self) -> 'AutoPipeline':
        """
        Return a copied AutoPipeline

        Returns:
            AutoPipeline: Copied AutoPipeline instance
        """
        return AutoPipeline(self.steps, self.explanations)
    

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
        return self.steps and hasattr(self.steps[-1][1], 'predict')
    
    def predict(self, X:pd.DataFrame, model_only:bool = False, **kwargs) -> list:
        """
        Run all the steps to predict labels from input data

        Args:
            X (pd.DataFrame): Features used as input of the pipeline
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
        
        return self.model.predict(X)

    def transform(self, X: pd.DataFrame) -> ndarray:
        for _, s in self.steps[:-1]: # ignore last step (training)
            X = s.transform(X)
        
        return X
    
    def add_explanation(
            self,
            step: 'Step',
            processings: list[str] = None,
            metrics: dict['Metric', float] = None,
            shap_values: list = None):
        """
        Explain a step of the pipeline.

        Args:
            step (Step): Step to explain.
            processings (list[str]): List of all the processings the
                                    the step has done to the data.
        """
        self.explanations.append(Explanation(step, processings, metrics, shap_values))
    
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

        explainer = shap.KernelExplainer(p, self.original_dataset or X)
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
