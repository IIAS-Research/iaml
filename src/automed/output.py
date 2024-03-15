"""
Output (and Input alias) is used to exchange data between Steps  
"""
from typing import TYPE_CHECKING

from copy import copy
import pandas as pd
import shap
import warnings

from .auto_pipeline import AutoPipeline
from .dataset import Dataset
from .explanation import Explanation


if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step


class Output:
    """
    Output of every Automed Step
    
    Attributes:
        dataset(Dataset): Dataset being built
        metrics(list[Metric]): List of metrics used to evaluate models
        main_metric(Metric): Main metric to evaluate model
        pipeline(AutoPipeline): Pipeline being built
        computed_metrics(dict): Results of metrics computation
        stacked_path(list): Stack of all steps used to build this Output
    """
    
    def __init__(self,
                dataset:Dataset=None,
                metrics:list['Metric']=None,
                auto_pipeline:'AutoPipeline'=None,
                stacked_path:list=None,
                main_metric:'Metric'=None):

        self.dataset = dataset
        self.metrics = copy(metrics) if metrics is not None else []
        self.pipeline = auto_pipeline or AutoPipeline() # Pipeline
        
        if main_metric is None and dataset and dataset.type_of_target:
            if 'continuous' in dataset.type_of_target:
                self.main_metric = 'r2_score'
            else:
                self.main_metric = 'balanced_accuracy'
        else:
            self.main_metric = main_metric
            
        self.computed_metrics = {}
        self.stacked_path = copy(stacked_path) if stacked_path is not None else []
        
    def add_stack(self, stack:'Step'):
        """
        Add a step to the stack

        Args:
            stack (step): Step to add
        """
        self.stacked_path.append(stack)
        
    def get_main_metric_value(self) -> float:
        """
        Get computed value of the main metric from saved metrics

        Returns:
            float: Main metric value
        """
        if self.computed_metrics and self.main_metric in self.computed_metrics:
            return self.computed_metrics[self.main_metric]
        return -1
    
    def __gt__(self, other:'Step'):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() > other.get_main_metric_value()
        if self.computed_metrics:
            return True
        if other.computed_metrics:
            return False
        
        return id(self) > id(other)
            
    def __lt__(self, other:'Step'):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() < other.get_main_metric_value()
        if self.computed_metrics:
            return False
        if other.computed_metrics:
            return True
        
        return id(self) < id(other)
            
    def __eq__(self, other:'Step'):
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() == other.get_main_metric_value()
        
        return id(self) == id(other)
        
    def to_output(self,
                dataset:Dataset=None,
                metrics:'Metric'=None,
                auto_pipeline:'AutoPipeline'=None) -> 'Output':
        """
        Create a copy of current instance and assign parameters values to attributes 

        Args:
            dataset (Dataset, optional): Replace current dataset. Defaults to None.
            metrics (Metric, optional): Replace current metrics. Defaults to None.
            auto_pipeline (AutoPipeline, optional): Replace current pipeline. Defaults to None.

        Returns:
            Output: New Output
        """
        return Output(
            dataset or self.dataset,
            metrics or copy(self.metrics),
            auto_pipeline or self.pipeline.copy(),
            stacked_path=self.stacked_path)
        
    def to_input(self,
                dataset:Dataset=None,
                metrics:'Metric'=None,
                auto_pipeline:'AutoPipeline'=None) -> 'Input':
        """
        Create a copy of current instance and assign parameters values to attributes 

        Args:
            dataset (Dataset, optional): Replace current dataset. Defaults to None.
            metrics (Metric, optional): Replace current metrics. Defaults to None.
            auto_pipeline (AutoPipeline, optional): Replace current pipeline. Defaults to None.

        Returns:
            Input: New Input
        """
        return Input(
            dataset or self.dataset,
            metrics or copy(self.metrics),
            auto_pipeline or self.pipeline.copy(),
            stacked_path=self.stacked_path)
    
    def add_transform(self, instance:'Step') -> 'Output':
        """
        Transforms the dataset using the provided function, and adds it to the
        stack of functions to be applied before prediction.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        self.dataset.transform(instance.transform)
            
        if self.pipeline is not None \
            and instance.transform is not None \
            and callable(instance.transform):
            self.pipeline.add_transform(instance)
        
        return self.to_output()
    
    
    def add_resample(self, instance:'Step') -> 'Output':
        """
        Resample the dataset using the resample method of provided instance.
        The method will be applied to the dataset just before training,
        after splitting into train and test.
        """
        self.dataset.resample(instance.resample)
        
        return self.to_output()
    
    def set_model(self, instance:'Step') -> 'Output':
        """
        Sets the resulting model of the pipeline to this output.
        Note: The function must be pickable and therefore must be named (not be
        a lambda) and be declared at the top level of a module.
        See https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled.
        """
        return self.to_output(auto_pipeline=self.pipeline.set_model(instance))
    
    def add_metric(self, metric:'Metric') -> None:
        """
        Add a new Metric to evaluate models

        Args:
            metric (Metric): metric to add
        """
        self.metrics.append(metric)
        
    def __str__(self) -> str:
        str_out = "[INPUT/OUTPUT]"
        if self.metrics:
            str_out = str_out + str(self.metrics) + " "
        if self.pipeline:
            str_out = str_out + str(self.pipeline) + " "
            
        return str_out
    
    def evaluate(self, dataset:Dataset, force:bool=False) -> dict:
        """
        Evaluate pipeline model with self.metrics on dataset
        If evaluate is called in training process, result will be cached in
        self.computed_metrics.

        Args:
            dataset (Dataset): Dataset used to compute metrics results
            force (bool, optional): If True & in training process,
                                    cache will not be use. Defaults to False.

        Returns:
            dict: Metric name as key and result as value
        """
        if not self.pipeline.have_model :
            return None
        
        training_stage = dataset.splitted # If dataset is splitted -> We here in the training stage
        
        if training_stage and not(force) and self.computed_metrics:
            return self.computed_metrics
            
        current_compute = {}
        for metric in self.metrics:
            result = dataset.compute_metric(self.pipeline, metric)
            current_compute[str(metric)] = result
            
        if training_stage:
            self.computed_metrics = current_compute
            
        return current_compute
    
    def explain_model(self, X: 'pd.DataFrame', max_evals: int = 'auto'):
        """
        Explains the model by computing SHAP values on the fitted model.
        Uses the train set as the masker, and the provided set as
        prediction.

        Args:
            X (DataFrame): Prediction set to compute SHAP values for.
            max_evals (int): Max number of SHAP evaluations to do on
                the dataset.

        Returns:
            Explanation: Model explanation, with an overview of the
                most important features, and graphs.
        """
        if not self.pipeline.have_model:
            raise RuntimeError('There is no model to explain.')

        step = self.pipeline.model

        def p(pred_data):
            return self.pipeline.predict_proba(pd.DataFrame(pred_data, columns=X.columns))[:, 1]

        # explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], med)
        n = 68
        explainer = shap.KernelExplainer(p, X)
        shap_values = explainer.shap_values(X, nsamples=n)

        import numpy as np
        shap_explanation = shap.Explanation(
            shap_values,
            base_values=np.tile(explainer.expected_value, (shap_values.shape[0], 1)),
            data=X,
            feature_names=X.columns)

        return Explanation(step, None, None, shap_explanation)
    
    def explain(self):
        """
        Explain all steps

        Returns:
            list: Explain strings
        """
        return list(map(lambda stack: stack.explain(), self.stacked_path))
    

class Input(Output):
    """
    Use as alias for Output
    """
