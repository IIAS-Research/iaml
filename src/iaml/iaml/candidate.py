"""Candidate is used to exchange data between Steps  """
from __future__ import annotations

import traceback
from typing import TYPE_CHECKING
from copy import copy, deepcopy
from hashlib import md5
import textwrap
import numpy as np
import pandas as pd
from .dataset import Dataset
from .cache import Cache
from .splitters import random_splitter
from .iaml_pipeline import IAMLPipeline
from .plot import MetricPlot
from .logger import Logger

if TYPE_CHECKING:
    from .metric import Metric
    from .step import Step


class Candidate:
    """Candidate of every IAML Step
    
    :param Dataset, optional dataset: Dataset being built. Default to None.
    :param list[Metric], optional metrics: List of metrics used to evaluate models. Default to None.
    :param IAMLPipeline, optional iaml_pipeline: Pipeline being built. Default to None.
    :param list, optional stacked_path: Stack of all steps used to build this Candidate.
        Default to None.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        metrics: list[Metric] = None,
        iaml_pipeline: IAMLPipeline = None,
        stacked_path: list = None,
        main_metric: 'Metric' = None) -> None:

        self.dataset = dataset
        """Dataset used for this candidate"""

        self.metrics = copy(metrics) if metrics is not None else []
        """List of metrics used to evaluate the model"""

        if iaml_pipeline is not None:
            self.pipeline = iaml_pipeline
        else:
            self.pipeline = IAMLPipeline(
                estimator_type=dataset.needed_estimator,
                original_dataset=dataset.X.copy()
            )

        if main_metric is None:
            if self.pipeline.estimator_type == "classifier":
                self.main_metric = 'balanced_accuracy'
            elif self.pipeline.estimator_type == "survival":
                self.main_metric = 'concordance_index_ipcw'
            else:
                self.main_metric = 'r2_score'
        else:
            self.main_metric = str(main_metric)

        self.computed_metrics = {}
        """Result dictionnary for all the metrics computed"""

        self.stacked_path = copy(stacked_path) if stacked_path is not None else []
        """Stack of all steps used to build this Candidate"""

        self.is_meta: bool = False
        """Whether it's a meta candidate or not"""

    def add_stack(self, stack: 'Step') -> None:
        """Add a step to the stack

        :param Step stack: Step to add
        """
        self.stacked_path.append(stack)

    def get_main_metric_value(self) -> float:
        """Get computed value of the main metric from saved metrics

        :return: Main metric value
        """
        if self.computed_metrics and self.main_metric in self.computed_metrics:
            return self.computed_metrics[self.main_metric]
        return -1

    def __gt__(self, other: 'Candidate') -> bool:
        """Check if a candidate is greater than another by various methods.
        
        - Main Metric Value
        - Presence of computed metrics
        - id of the candidate
        
        :param Candidate other: The other Candidate to compare to.
        :return: Greater than?
        """
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() > other.get_main_metric_value()
        if self.computed_metrics:
            return True
        if other.computed_metrics:
            return False

        return id(self) > id(other)

    def __lt__(self, other: 'Candidate'):
        """Check if a candidate is less than another by various methods.
        
        - Main Metric Value
        - Presence of computed metrics
        - id of the candidate
        
        :param Candidate other: The other Candidate to compare to.
        :return: Less than?
        """
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() < other.get_main_metric_value()
        if self.computed_metrics:
            return False
        if other.computed_metrics:
            return True

        return id(self) < id(other)

    def __eq__(self, other: 'Candidate'):
        """Check if a candidate is equal to another by various methods.
        
        - Main Metric Value
        - Presence of computed metrics
        - id of the candidate
        
        :param Candidate other: The other Candidate to compare to.
        :return: Equal to?
        """
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_value() == other.get_main_metric_value()

        return id(self) == id(other)

    def to_output(
        self,
        dataset: Dataset = None,
        metrics: list[Metric] = None,
        iaml_pipeline: IAMLPipeline = None) -> 'Candidate':
        """Create a copy of current instance and assign parameters values to attributes 

        :param Dataset, optional dataset: Replace current dataset. Defaults to None.
        :param Metric, optional metrics: Replace current metrics. Defaults to None.
        :param IAMLPipeline, optional iaml_pipeline: Replace current pipeline. Defaults to None.
        :return: New Candidate
        """
        if iaml_pipeline is None:
            iaml_pipeline = self.pipeline.copy()

        return Candidate(
            dataset or deepcopy(self.dataset),
            metrics or copy(self.metrics),
            iaml_pipeline=iaml_pipeline,
            stacked_path=self.stacked_path)

    def to_input(self,
                dataset: Dataset = None,
                metrics: list[Metric] = None,
                iaml_pipeline: IAMLPipeline = None) -> 'Candidate':
        """Create a copy of current instance and assign parameters values to attributes 

        :param Dataset, optional dataset: Replace current dataset. Defaults to None.
        :param Metric, optional metrics: Replace current metrics. Defaults to None.
        :param IAMLPipeline, optional iaml_pipeline: Replace current pipeline. Defaults to None.
        :return: New Candidate
        """
        return self.to_output(dataset, metrics, iaml_pipeline)

    def add_to_pipeline(self, instance: 'Step') -> 'Candidate':
        """Add a Step to prediction Pipeline.
        instance must implement one of these methods :
            - transform(X) : Apply column transformations to Dataset.
            - predict(X) : Predict values with AI model.
            - resample(X,y) : Apply row transformations to Dataset (will be run just 
                before prediction).
        
        :param Step instance: Add a step to the pipeline.
        :return: New Candidate
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

    def add_metric(self, metric: 'Metric') -> None:
        """Add a new Metric to evaluate models

        :param Metric metric: metric to add
        """
        self.metrics.append(metric)

    def __str__(self) -> str:
        """String representation of the Candidate
        
        :return: String representatio n of the candidate.
        """
        name = [name for name, _ in self.pipeline.steps]
        main_metric = self.get_main_metric_value()
        if main_metric:
            name = f"{main_metric} : {name}"
        return name

    def training_evaluate(
        self,
        dataset: Dataset,
        splitter: callable = random_splitter) -> dict:
        """Evaluate pipeline model with self.metrics on dataset
        If evaluate is called in training process, result will be cached in
        self.computed_metrics.

        :param Dataset dataset: Dataset used to compute metrics results
        :param callable, optional splitter: The split method to be used. Default to random_splitter.
        :return: Metric name as key and result as value
        """
        if not self.pipeline.have_model:
            return None
        metrics: list[dict] = []

        # without cache !
        from_cache: bool = True
        to_cache: list = True
        splitted_datasets = Cache().from_cache(self.fingerprint(), dataset.X)
        if not splitted_datasets or self.is_meta: # Cannot use cache with meta for now
            from_cache = False
            to_cache: list = []
            splitted_datasets: list[tuple[Dataset, Dataset]] = splitter(dataset)

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
                metrics.append(self.__compute_metrics(
                    test_ds.X,
                    test_ds.y,
                    pipeline=copied_pipe,
                    X_train=train_ds.X,
                    y_train=train_ds.y,
                    model_only= not self.is_meta)
                    )
                if not from_cache:
                    to_cache.append((train_ds, test_ds))
            except ValueError:
                return {}

        if not from_cache:
            Cache().add_to_cache(self.fingerprint(), dataset.X, to_cache)

        self.computed_metrics = { k: np.mean([ metric[k] or 0 for metric in metrics ]) \
            for k in map(str, self.metrics) }

        return self.computed_metrics

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Evaluate pipeline performances with self.metrics

        :param pd.DataFrame X: Features.
        :param np.ndarray y: label.
        :return: Computed metrics.
        """
        if not self.pipeline.have_model:
            return None

        return self.__compute_metrics(X, np.array(y))

    def __compute_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        pipeline : IAMLPipeline = None,
        X_train: pd.DataFrame = None,
        y_train: np.ndarray = None,
        **kwargs) -> dict:
        """Perform metrics computation
        
        :param pd.DataFrame X_test: The dataframe to compute metrics on.
        :param np.ndarray y_test: The dataframe labels used for evaluation.
        :param IAMLPipeline, optional: The pipeline to use for evaludation. Default to None.
        :param pd.DataFrame, optional X_train: DataFrame used for metrics comparison. 
            Default to None.
        :param np.ndarray, optional y_train: DataFrame labels used for metrics comparison.
            Default to None.
        :param dict, optional \\**kwargs: Additional parameters.
        :return: computed metrics
        """
        if pipeline is None: # Is no pipeline in args -> Use the main one
            pipeline = self.pipeline

        X_test = X_test.copy(deep=True)
        y_test = deepcopy(y_test)

        if y_train is None or X_train is None:
            y_train=self.dataset.y
            X_train=self.dataset.X

        computed = {}
        needs = {metric.needed_prediction for metric in self.metrics}

        for need in needs:
            try:
                method = getattr(pipeline, need)
                y_pred = method(X_test, **kwargs)
                for metric in self.metrics:
                    try:
                        if metric.needed_prediction == need:
                            computed[str(metric)] = metric.compute(
                                y_test,
                                y_pred,
                                y_train=y_train,
                                X_train=X_train
                            )
                    except Exception:  # pylint: disable=broad-exception-caught
                        Logger().error(traceback.format_exc())
            except AttributeError:
                pass
        return computed

    def __metric_value(self, metric: Metric) -> float | None:
        """Get a specific metric value
        
        :param Metric metric: The metric we want to get the value.
        :return: The metric value or None if this Metric doesn't exsists.
        """
        for key, value in self.computed_metrics.items():
            if key == str(metric):
                return value
        return None

    def bibliography(self, structured: bool = False) -> str | list[dict]:
        """Format a string with all step's referencies

        :param bool, optional structured: Str or JSON serializable bibliography. Default to Str.
        :return: Formatted bibliography or structured bibliography.
        """
        return self.pipeline.bibliography(structured)

    def fingerprint(self) -> str:
        """Generate a fingerprint to identify this instance

        :return: String fingerprint
        """
        to_hash = "\n".join([step.fingerprint() \
                for _, step in [*self.pipeline.transformers, *self.pipeline.resamplers]])

        return md5(to_hash.encode()).hexdigest()

    def explain(self) -> list:
        """Explain all steps

        :return: list of Explain strings
        """
        # Results explain
        metrics = '\n'.join([
            f'| `{m}` | **{self.__metric_value(m):.4f}** | *{m.explain()}* |'
            for m in self.metrics
        ])
        results_explain: str = textwrap.dedent(f'''
            ### Results
            | Metric name | Computed value | Description |
            | ----------- | -------------- | ----------- |
            {metrics}
            ''') if len(metrics) > 0 else ""

        return [*self.pipeline.explanations, results_explain]

    def explain_model_performance(
        self,
        X_test: pd.DataFrame,
        y_test: list,
        X_train: pd.DataFrame = None,
        y_train: list = None,
        **kwargs) -> list[MetricPlot]:
        """Return a list of plot that explain models performances

        :param pd.DataFrame X_test: Features.
        :param list y_test: Target.
        :param pd.DataFrame, optional X_train: Train features. Default to None.
        :param list, optional y_train: Train target. Default to None.
        :param dict, optional \\**kwargs: Additional Parameters.
        :return: List of plot instances.
        """
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test[y_test.columns[0]]
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train[y_train.columns[0]]


        plots = []
        for plot_sub_class in MetricPlot.__subclasses__():
            # Verify if a subclass is suitable or not
            if plot_sub_class.suitable(self.dataset.type_of_target):
                plot = plot_sub_class(self.pipeline,
                    deepcopy(X_test), deepcopy(y_test),
                    X_train=deepcopy(X_train), y_train=deepcopy(y_train), **kwargs)
                plots.append(plot)

        return plots
