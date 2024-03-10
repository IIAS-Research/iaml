"""
Output (and Input alias) is used to exchange data between Steps  
"""
from copy import copy
from typing import TYPE_CHECKING
from .dataset import Dataset

from .auto_pipeline import AutoPipeline


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
        
    def add_to_pipeline(self, instance:'Step') -> 'Output':
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
                self.dataset.resample(instance.resample)
                
        return self.to_output()
        
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
