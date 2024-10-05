"""
Candidate is used to exchange data between Steps  
"""
from typing import TYPE_CHECKING, List, Dict
from copy import copy, deepcopy
from hashlib import md5
import numpy as np
import pandas as pd
from .dataset import Dataset
from .cache import Cache
from .splitter import random_splitter
from .iaml_pipeline import IAMLPipeline
if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step


class Candidate:
    """
    Candidate of every Automed Step
    
    Attributes:
        dataset(Dataset): Dataset being built
        metrics(list[Metric]): List of metrics used to evaluate models
        main_metric(Metric): Main metric to evaluate model
        pipeline(IAMLPipeline): Pipeline being built
        computed_metrics(dict): Results of metrics computation
        stacked_path(list): Stack of all steps used to build this Candidate
    """
    
    def __init__(self,
                dataset:Dataset=None,
                metrics:list['Metric']=None,
                iaml_pipeline:'IAMLPipeline'=None,
                stacked_path:list=None,
                main_metric:'Metric'=None):

        self.dataset = dataset
        self.metrics = copy(metrics) if metrics is not None else []
        self.pipeline = iaml_pipeline \
            or IAMLPipeline(
                estimator_type=dataset.needed_estimator,
                original_dataset=dataset.X.copy()
            )
        
        if main_metric is None:
            if self.pipeline.estimator_type == "classifier":
                self.main_metric = 'balanced_accuracy'
            elif self.pipeline.estimator_type == "survival":
                self.main_metric = 'concordance_index'
            else:
                self.main_metric = 'r2_score'
        else:
            self.main_metric = str(main_metric)
            
        self.computed_metrics = {}
        self.stacked_path = copy(stacked_path) if stacked_path is not None else []
        self.is_meta:bool = False
        
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
                iaml_pipeline:'IAMLPipeline'=None) -> 'Candidate':
        """
        Create a copy of current instance and assign parameters values to attributes 

        Args:
            dataset (Dataset, optional): Replace current dataset. Defaults to None.
            metrics (Metric, optional): Replace current metrics. Defaults to None.
            iaml_pipeline (IAMLPipeline, optional): Replace current pipeline. Defaults to None.

        Returns:
            Candidate: New Candidate
        """
        return Candidate(
            dataset or deepcopy(self.dataset),
            metrics or copy(self.metrics),
            iaml_pipeline or self.pipeline.copy(),
            stacked_path=self.stacked_path)
        
    def to_input(self,
                dataset:Dataset=None,
                metrics:'Metric'=None,
                iaml_pipeline:'IAMLPipeline'=None) -> 'Candidate':
        """
        Create a copy of current instance and assign parameters values to attributes 

        Args:
            dataset (Dataset, optional): Replace current dataset. Defaults to None.
            metrics (Metric, optional): Replace current metrics. Defaults to None.
            iaml_pipeline (IAMLPipeline, optional): Replace current pipeline. Defaults to None.

        Returns:
            Candidate: New Candidate
        """
        return self.to_output(dataset, metrics, iaml_pipeline)
        
    def add_to_pipeline(self, instance:'Step') -> 'Candidate':
        """
        Add a Step to prediction Pipeline.
        instance must implement one of these methods :
            - transform(X) : Apply column transformations to Dataset
            - predict(X) : Predict values with AI model
            - resample(X,y) : Apply row transformations to Dataset 
                (will be run just before prediction)
        """
        if self.pipeline is not None:
            if hasattr(instance, 'transform') and callable(instance.transform):
                self.dataset.transform(instance.transform)
                self.pipeline.add_transform(instance)
            elif hasattr(instance, 'predict') and callable(instance.predict):
                self.pipeline.set_model(instance)
            elif hasattr(instance, 'resample') and callable(instance.resample):
                self.pipeline.add_resample(instance)
                # Resample in Dataset used in pipeline generation step
                self.dataset = self.dataset.resample(instance.resample)
                
        return self.to_output()
        
    def add_metric(self, metric:'Metric') -> None:
        """
        Add a new Metric to evaluate models

        Args:
            metric (Metric): metric to add
        """
        self.metrics.append(metric)
        
    def __str__(self) -> str:
        name = [name for name, _ in self.pipeline.steps]
        main_metric = self.get_main_metric_value()
        if main_metric:
            name = f"{main_metric} : {name}"
        return name
    
    def training_evaluate(self, dataset:Dataset, \
                splitter:callable=random_splitter) -> dict:
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
        if not self.pipeline.have_model:
            return None
        metrics:list[dict] = []
        
        # without cache !
        from_cache:bool = True
        to_cache:list = True
        splitted_datasets = Cache().from_cache(self.fingerprint(), dataset.X)
        if not splitted_datasets or self.is_meta: # Cannot use cache with meta for now
            from_cache = False
            to_cache:list = []
            splitted_datasets:list[tuple[Dataset, Dataset]] = splitter(dataset)

        for train_ds, test_ds in splitted_datasets:
            copied_pipe = deepcopy(self.pipeline)
            
            if self.is_meta:
                copied_pipe.fit(train_ds.X, train_ds.y)
            else:
                # Fit in two step to allow caching
                if not from_cache:
                    train_ds =  train_ds.decline(*copied_pipe.fit_transform(train_ds.X, train_ds.y))
                    test_ds =  test_ds.decline(copied_pipe.transform(test_ds.X), test_ds.y)
                    
                copied_pipe.fit(train_ds.X, train_ds.y, only_predictor=True)
                
            try:
                metrics.append(self.__compute_metrics(test_ds.X, test_ds.y, pipeline=copied_pipe, model_only= not self.is_meta))
                if not from_cache:
                    to_cache.append((train_ds, test_ds))
            except ValueError:
                return {}
        
        if not from_cache:
            Cache().add_to_cache(self.fingerprint(), dataset.X, to_cache)    
        
        self.computed_metrics = { k: np.mean([ metric[k] or 0 for metric in metrics ]) \
            for k in map(str, self.metrics) }
        
        return self.computed_metrics
    
    
    def evaluate(self, X:pd.DataFrame, y:np.array) -> dict:
        """
        Evaluate pipeline performances with self.metrics

        Args:
            X (pd.DataFrame): Features
            y (np.array): label

        Returns:
            dict: Computed metrics
        """
        if not self.pipeline.have_model:
            return None
        
        return self.__compute_metrics(X, np.array(y))
    
        
    def __compute_metrics(self, X_test:pd.DataFrame, y_test:np.array, pipeline=None, **kwargs) -> dict:
        
        if pipeline is None: # Is no pipeline in args -> Use the main one
            pipeline = self.pipeline
            
        X_test = X_test.copy(deep=True)
        y_test = deepcopy(y_test)
            
        computed = {}
        needs = {metric.needed_prediction for metric in self.metrics}
        
        for need in needs:
            try:
                method = getattr(pipeline, need)
                y_pred = method(X_test, **kwargs)
                for metric in self.metrics:
                    if metric.needed_prediction == need:
                        computed[str(metric)] = metric.compute(
                            y_test,
                            y_pred,
                            y_train=self.dataset.y,
                            X_train=self.dataset.X
                        )
            except AttributeError:
                pass

        return computed
    
    def __metric_value(self, metric) -> float:
        for key, value in self.computed_metrics.items():
            if key == str(metric):
                return value
        return None
    
    def bibliography(self, structured: bool=False) -> str | List[Dict]:
        """Format a string with all step's referencies

        Returns:
            str | List[Didct]: formatted bibliography or structured bibliography
        """
        return self.pipeline.bibliography(structured)
    
        
    def fingerprint(self) -> str:
        """
        Generate a fingerprint to identify this instance

        Returns:
            str: String fingerprint
        """
        to_hash = "\n".join([step.fingerprint() \
                for _, step in [*self.pipeline.transformers, *self.pipeline.resamplers]])
        
        return md5(to_hash.encode()).hexdigest()
    
    def explain(self) -> list:
        """
        Explain all steps

        Returns:
            list: Explain strings
        """
        # Results explain 
        metrics = '\n'.join([
            f'| `{m}` | **{self.__metric_value(m):.4f}** | *{m.explain()}* |'
            for m in self.metrics
        ])
        results_explain:str = f'''
### Results
| Metric name | Computed value | Description |
| ----------- | -------------- | ----------- |
{metrics}
''' if len(metrics) > 0 else ""
        
        return [*self.pipeline.explanations, results_explain]
    
