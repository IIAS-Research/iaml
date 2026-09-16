"""Candidate is used to exchange data between Steps  """
from __future__ import annotations

import traceback
import os
import time
from typing import TYPE_CHECKING, Any
from copy import copy, deepcopy
from hashlib import md5
import textwrap
import numpy as np
import pandas as pd
from .dataset import Dataset
from .cache import Cache
from .cache_keys import hash_evaluation_context
from .splitters import random_splitter
from .iaml_pipeline import IAMLPipeline
from .metric_plot import MetricPlot
from .logger import Logger
from .step_cache import StepCache

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

        self.dataset: Dataset = dataset
        """Dataset used for this candidate"""

        self.metrics: list[Metric] = copy(metrics) if metrics is not None else []
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

        self.computed_metrics: dict = {}
        """Result dictionnary for all the metrics computed"""

        self.fold_metrics: list[dict[str, Any]] = []
        """Per-fold metrics computed during the latest internal cross-validation."""

        self.training_audit: dict[str, Any] | None = None
        """Structured audit payload for the latest training evaluation."""

        self.stacked_path: list = copy(stacked_path) if stacked_path is not None else []
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
        logger = None
        diag_enabled = os.environ.get("IAML_DIAG", "").lower() in ["1", "true", "yes"]
        if diag_enabled:
            from .logger import Logger  # pylint: disable=import-outside-toplevel
            logger = Logger()
            if logger.verbose <= 1:
                logger = None

        pipeline_copy_time = 0.0
        dataset_copy_time = 0.0
        if iaml_pipeline is None:
            start = time.perf_counter()
            iaml_pipeline = self.pipeline.copy()
            pipeline_copy_time = time.perf_counter() - start

        if dataset is None:
            start = time.perf_counter()
            dataset = deepcopy(self.dataset)
            dataset_copy_time = time.perf_counter() - start

        if logger is not None:
            try:
                steps = self.pipeline.training_steps
                step_count = len(steps)
                cache_count = 0
                step_cache = StepCache()
                for _, step in steps:
                    if hasattr(step, "_cache_id"):
                        cache_count += step_cache.size_for_step(step._cache_id)
                    elif hasattr(step, "caches") and step.caches is not None:
                        cache_count += len(step.caches)
                candidate_refs = 0
                for _, step in steps:
                    if hasattr(step, "candidate") and step.candidate is not None:
                        if isinstance(step.candidate, list):
                            candidate_refs += len(step.candidate)
                        else:
                            candidate_refs += 1
            except Exception:  # pylint: disable=broad-except
                step_count = None
                cache_count = None
                candidate_refs = None

            logger.info(
                "diag: to_output copy pipeline=%.3fs dataset=%.3fs steps=%s caches=%s candidates=%s",
                pipeline_copy_time,
                dataset_copy_time,
                step_count,
                cache_count,
                candidate_refs,
            )

        return Candidate(
            dataset,
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

    @classmethod
    def __serialize_audit_value(cls, value: Any) -> Any:
        """Convert runtime values into JSON-friendly audit payloads."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(key): cls.__serialize_audit_value(current)
                for key, current in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls.__serialize_audit_value(current) for current in value]
        if callable(value):
            return getattr(value, "__name__", str(value))
        return str(value)

    def __serialize_metric_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Convert metric outputs into a stable, serializable format."""
        return {
            str(name): self.__serialize_audit_value(value)
            for name, value in values.items()
        }

    def __aggregate_metrics(self, fold_results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate fold-level metric dictionaries into global scores."""
        computed_metrics: dict[str, float] = {}
        for metric in self.metrics:
            name = str(metric)
            values = [
                metric_values.get(name)
                for metric_values in fold_results
                if name in metric_values
            ]
            if values:
                computed_metrics[name] = float(np.mean([
                    value if value is not None else 0 for value in values
                ]))
            else:
                computed_metrics[name] = 0
        return computed_metrics

    def pipeline_audit_summary(self) -> dict[str, Any]:
        """Return a serializable summary of the pipeline steps and their config."""
        summarized_steps: list[dict[str, Any]] = []
        groups = [
            ("resampler", self.pipeline.resamplers),
            ("transformer", self.pipeline.transformers),
            ("predictor", [self.pipeline.predictor] if self.pipeline.predictor else []),
        ]

        for role, steps in groups:
            for name, step in steps:
                summarized_steps.append(
                    {
                        "role": role,
                        "name": name,
                        "class": step.__class__.__name__,
                        "tags": sorted(step.tags) if step.tags else [],
                        "configuration": self.__serialize_audit_value(
                            step.resume_configuration()
                        ),
                    }
                )

        return {
            "fingerprint": self.pipeline.fingerprint(),
            "estimator_type": self.pipeline.estimator_type,
            "steps": summarized_steps,
        }

    def build_training_audit(
        self,
        dataset: Dataset,
        fold_metrics: list[dict[str, Any]],
        aggregated_metrics: dict[str, Any],
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build a structured record for later audit on the IAML object."""
        return {
            "pipeline_fingerprint": self.pipeline.fingerprint(),
            "dataset_fingerprint": dataset.fingerprint(),
            "dataset_shape": {
                "rows": int(dataset.X.shape[0]),
                "columns": int(dataset.X.shape[1]),
            },
            "main_metric": self.main_metric,
            "status": status,
            "error": error,
            "metrics": self.__serialize_metric_values(aggregated_metrics),
            "fold_metrics": deepcopy(fold_metrics),
            "pipeline": self.pipeline_audit_summary(),
        }

    def training_evaluate(
        self,
        dataset: Dataset,
        splitter: callable = random_splitter,
        cache_split: bool = True,
        store_audit: bool = False) -> dict:
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
        fold_metrics: list[dict[str, Any]] = []
        self.fold_metrics = []
        self.training_audit = None

        splitter_fingerprint = (
            hash_evaluation_context(splitter) if cache_split and not self.is_meta else None
        )
        cache_key = (
            f"splits_{self.fingerprint()}_{splitter_fingerprint}"
            if splitter_fingerprint is not None else None
        )
        dataset_key = dataset.fingerprint() if cache_key else None
        from_cache = False
        to_cache: list[tuple[Dataset, Dataset]] = []
        splitted_datasets = Cache().from_cache(cache_key, dataset_key) if cache_key else None
        if splitted_datasets:
            from_cache = True
        else:
            splitted_datasets = splitter(dataset)

        for fold_index, (train_ds, test_ds) in enumerate(splitted_datasets, start=1):
            copied_pipe = deepcopy(self.pipeline)

            try:
                if self.is_meta:
                    copied_pipe.fit(train_ds.X, train_ds.y)
                else:
                    # Fit in two step to allow caching
                    if not from_cache:
                        train_ds = train_ds.decline(*copied_pipe.fit_transform(train_ds.X, train_ds.y))
                        test_ds = test_ds.decline(copied_pipe.transform(test_ds.X), test_ds.y)

                    copied_pipe.fit(train_ds.X, train_ds.y, only_predictor=True)
            except (ValueError, np.linalg.LinAlgError) as exc:
                Logger().warning(
                    f"Skip candidate {self._pipeline_signature()} after training failure: {exc!r}"
                )
                if store_audit:
                    self.training_audit = self.build_training_audit(
                        dataset=dataset,
                        fold_metrics=fold_metrics,
                        aggregated_metrics=self.__aggregate_metrics(metrics),
                        status="failed",
                        error=f"training failure: {exc!r}",
                    )
                return {}

            try:
                fold_result = self.__compute_metrics(
                    test_ds.X,
                    test_ds.y,
                    pipeline=copied_pipe,
                    X_train=train_ds.X,
                    y_train=train_ds.y,
                    model_only= not self.is_meta)
                metrics.append(fold_result)
                if store_audit:
                    fold_metrics.append(
                        {
                            "fold": fold_index,
                            "train_shape": {
                                "rows": int(train_ds.X.shape[0]),
                                "columns": int(train_ds.X.shape[1]),
                            },
                            "test_shape": {
                                "rows": int(test_ds.X.shape[0]),
                                "columns": int(test_ds.X.shape[1]),
                            },
                            "metrics": self.__serialize_metric_values(fold_result),
                        }
                    )
                if cache_key and not from_cache:
                    to_cache.append((train_ds, test_ds))
            except ValueError as exc:
                Logger().warning(
                    f"Skip candidate {self._pipeline_signature()} after metric computation failure "
                    f"(predict/predict_proba raised ValueError: {exc!r})"
                )
                if store_audit:
                    self.training_audit = self.build_training_audit(
                        dataset=dataset,
                        fold_metrics=fold_metrics,
                        aggregated_metrics=self.__aggregate_metrics(metrics),
                        status="failed",
                        error=f"metric failure: {exc!r}",
                    )
                return {}

        if cache_key and not from_cache:
            Cache().add_to_cache(cache_key, dataset_key, to_cache)

        computed_metrics = self.__aggregate_metrics(metrics)
        self.computed_metrics = computed_metrics
        if store_audit:
            self.fold_metrics = deepcopy(fold_metrics)
            self.training_audit = self.build_training_audit(
                dataset=dataset,
                fold_metrics=fold_metrics,
                aggregated_metrics=computed_metrics,
                status="success",
            )

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
                Logger().warning(
                    f"Pipeline {self._pipeline_signature()} does not expose '{need}' needed by metrics."
                )
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
        if hasattr(self.pipeline, "transformers_resamplers_fingerprint"):
            return self.pipeline.transformers_resamplers_fingerprint()

        to_hash = "\n".join([
            step.fingerprint()
            for _, step in [*self.pipeline.transformers, *self.pipeline.resamplers]
        ])
        return md5(to_hash.encode()).hexdigest()

    def _pipeline_signature(self) -> str:
        """Return a short, logging-safe pipeline identifier."""
        try:
            names = [name for name, _ in self.pipeline.training_steps if name]
            if names:
                return " -> ".join(names)
        except Exception:
            pass
        return getattr(self.pipeline, "name", self.pipeline.__class__.__name__)

    def describe_metrics(self) -> str:
        """Explain all metrics

        :return: Markdown table of all metrics.
        """
        metrics = '\n            '.join([
            f'| `{m}` | **{self.__metric_value(m):.4f}** | *{m.explain()}* |'
            for m in self.metrics
        ])

        return textwrap.dedent(f'''\
            ### Results
            | Metric name | Computed value | Description |
            | ----------- | -------------- | ----------- |
            {metrics}
            ''') if len(metrics) > 0 else ""

    def describe_steps(self) -> list[str]:
        """Explain all steps

        :return: List of explanation strings for each step.
        """
        return self.pipeline.explanations

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
                plot = plot_sub_class().compute(self.pipeline,
                    deepcopy(X_test), deepcopy(y_test),
                    X_train=deepcopy(X_train), y_train=deepcopy(y_train), **kwargs)
                plots.append(plot)

        return plots

    def explain_feature_importance(self, X: pd.DataFrame, nsamples: int = 20):
        """
        Explains the model by computing SHAP values on the fitted model. Uses
        the train set as the masker, and the provided set as prediction.

        :param pd.DataFrame X: Prediction set to compute SHAP values for.
        :param int, optional nsamples: Number of samples to pick from the masker to pick feature 
            data from for each row in the provided prediction dataset. More samples means more 
            accurate SHAP values and longer computing times. Defaults to 20.
        :return: Model explanation, with an overview of the most important features, and graphs.
        """
        return self.pipeline.explain_model(X, nsamples)

    def predict(self, X: pd.DataFrame) -> list:
        """Run all the steps to predict labels from candidate data

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :return: Predicted values
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> list:
        """Run all the steps to predict labels from candidate data

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :return: Predicted values
        """
        return self.pipeline.predict(X)
