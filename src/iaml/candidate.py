"""Candidate is used to exchange data between Steps  """
from __future__ import annotations

import os
import time
from math import isfinite
from numbers import Real
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
from .study_analyses import (
    CompiledAnalysis, analyses_signature,
)

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
    :param main_metric: Metric or metric name used to rank candidates.
        If None, use the default metric for the task.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        metrics: list[Metric] = None,
        iaml_pipeline: IAMLPipeline = None,
        stacked_path: list = None,
        main_metric: Metric | str | None = None,
        *,
        metric_definitions: list[CompiledAnalysis] | None = None,
        study_snapshot: Any = None) -> None:

        self.dataset: Dataset = dataset
        """Dataset used for this candidate"""

        self.metrics: list[Metric] = copy(metrics) if metrics is not None else []
        """List of metrics used to evaluate the model"""

        self.metric_definitions = deepcopy(metric_definitions)
        if self.metric_definitions is not None:
            self.metrics = [definition.component for definition in self.metric_definitions]
        self.study_snapshot = deepcopy(study_snapshot)
        self.metric_report: list[dict[str, Any]] = []
        self.evaluation_report: list[dict[str, Any]] = []
        self.metric_coverage: dict[str, dict[str, Any]] = {}
        self._legacy_metric_ids = tuple(id(metric) for metric in self.metrics)
        self._legacy_metrics_signature = hash_evaluation_context(self.metrics)

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
            self.main_metric = main_metric

        self.computed_metrics: dict = {}
        """Result dictionnary for all the metrics computed"""

        self.fold_metrics: list[dict[str, Any]] = []
        """Per-fold metrics computed during the latest internal cross-validation."""

        self.training_audit: dict[str, Any] | None = None
        """Structured audit payload for the latest training evaluation."""

        self.stacked_path: list = copy(stacked_path) if stacked_path is not None else []
        """Stack of all steps used to build this Candidate"""

    def add_stack(self, stack: 'Step') -> None:
        """Add a step to the stack

        :param Step stack: Step to add
        """
        self.stacked_path.append(stack)

    @property
    def main_metric(self) -> str:
        """Name of the metric used to rank candidates."""
        return str(self._main_metric)

    @main_metric.setter
    def main_metric(self, metric: Metric | str) -> None:
        self._main_metric = metric

    def get_main_metric(self) -> Metric | str:
        """Return the metric definition, or its name when no definition is available."""
        if isinstance(self._main_metric, str):
            return next(
                (metric for key, metric in self._metric_items() if key == self.main_metric),
                self._main_metric,
            )
        return self._main_metric

    def _metric_items(self):
        """Pair runtime metrics with their frozen public result keys."""
        if self.metric_definitions is not None:
            return [(definition.key, metric) for definition, metric in
                    zip(self.metric_definitions, self.metrics)]
        return [(str(metric), metric) for metric in self.metrics]

    def metrics_signature(self) -> str | None:
        """Identify metric configuration before stateful computations mutate it."""
        if self.metric_definitions is not None:
            return analyses_signature(self.metric_definitions)
        identities = tuple(id(metric) for metric in self.metrics)
        if identities != self._legacy_metric_ids:
            self._legacy_metric_ids = identities
            self._legacy_metrics_signature = hash_evaluation_context(self.metrics)
        return self._legacy_metrics_signature

    def evaluation_context_signature(self) -> str | None:
        """Return the fixed metric and objective context used by score caches."""
        signature = self.metrics_signature()
        return hash_evaluation_context(signature, self.main_metric) if signature is not None else None

    def get_main_metric_value(self) -> float:
        """Get computed value of the main metric from saved metrics

        :return: Main metric value
        """
        if self.computed_metrics and self.main_metric in self.computed_metrics:
            return self.computed_metrics[self.main_metric]
        return -1

    def get_main_metric_score(self) -> float:
        """Return a ranking score where higher is better, preserving raw metrics.

        Metric names are resolved against the candidate's evaluation metrics.
        A candidate without its main metric always ranks below an evaluated one.
        """
        if self.main_metric not in self.computed_metrics:
            return float('-inf')
        metric = self.get_main_metric()
        value = self.get_main_metric_value()
        return value if getattr(metric, 'greater_is_better', True) else -value

    def __gt__(self, other: 'Candidate') -> bool:
        """Check if a candidate is greater than another by various methods.
        
        - Main Metric Value
        - Presence of computed metrics
        - id of the candidate
        
        :param Candidate other: The other Candidate to compare to.
        :return: Greater than?
        """
        if self.computed_metrics and other.computed_metrics:
            return self.get_main_metric_score() > other.get_main_metric_score()
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
            return self.get_main_metric_score() < other.get_main_metric_score()
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
            return self.get_main_metric_score() == other.get_main_metric_score()

        return id(self) == id(other)

    def to_output(
        self,
        dataset: Dataset = None,
        metrics: list[Metric] = None,
        iaml_pipeline: IAMLPipeline = None) -> 'Candidate':
        """Create a copy of the current instance, preserving its ranking metric.

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

        copied = Candidate(
            dataset,
            copy(self.metrics) if metrics is None else metrics,
            iaml_pipeline=iaml_pipeline,
            stacked_path=self.stacked_path,
            main_metric=self._main_metric,
            metric_definitions=self.metric_definitions if metrics is None else None,
            study_snapshot=self.study_snapshot)
        if metrics is None:
            copied._legacy_metrics_signature = self.metrics_signature()
        return copied

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

        The instance must implement one of these methods:

        - ``transform(X)``: Apply column transformations to the dataset.
        - ``predict(X)``: Predict values with an AI model.
        - ``resample(X, y)``: Apply row transformations to the training dataset.
        
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
        if self.metric_definitions is not None:
            from .study_analyses import _definition
            definition = _definition(deepcopy(metric), kind="metrics")
            if definition.key in {key for key, _ in self._metric_items()}:
                raise ValueError(f"Metric result key '{definition.key}' already exists")
            self.metric_definitions.append(definition)
            self.metrics.append(definition.component)
        else:
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

    def __valid_main_metric(self, scores: dict[str, Any]) -> bool:
        """A main metric must be a finite numeric scalar; zero remains valid."""
        return self.__finite_metric(scores.get(self.main_metric))

    @staticmethod
    def __finite_metric(value: Any) -> bool:
        """Validate one scalar without coercing arrays or missing values."""
        try:
            return isinstance(value, Real) and isfinite(value)
        except (TypeError, ValueError, OverflowError):
            return False

    def __aggregate_metrics(self, fold_results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate fold-level metric dictionaries into global scores."""
        computed_metrics: dict[str, float] = {}
        self.metric_coverage = {}
        for name, _ in self._metric_items():
            values = [
                metric_values[name]
                for metric_values in fold_results
                if name in metric_values and self.__finite_metric(metric_values[name])
            ]
            self.metric_coverage[name] = {
                "available": len(values), "total": len(fold_results),
                "values": deepcopy(values), "complete": len(values) == len(fold_results),
            }
            if values and len(values) == len(fold_results):
                computed_metrics[name] = float(np.mean(values))
        return computed_metrics

    def pipeline_audit_summary(self) -> dict[str, Any]:
        """Return a serializable summary of the pipeline steps and their config."""
        summarized_steps: list[dict[str, Any]] = []
        for name, step in self.pipeline.training_steps:
            if self.pipeline.predictor is not None and step is self.pipeline.predictor[1]:
                role = "predictor"
            elif callable(getattr(step, 'transform', None)):
                role = "transformer"
            else:
                role = "resampler"
            summarized_steps.append(
                {
                    "role": role,
                    "name": name,
                    "class": step.__class__.__name__,
                    "tags": sorted(step.tags) if step.tags else [],
                    "configuration": self.__serialize_audit_value(
                        step.resume_configuration()
                    ),
                    "alias": getattr(step, "_flow_alias", None),
                    "node_id": getattr(step, "_flow_node_id", None),
                    "variant_id": getattr(step, "_flow_variant_id", None),
                    "choice_id": getattr(step, "_flow_choice_id", None),
                    "search_policy": self.__serialize_audit_value({
                        "parameters": getattr(step, "_flow_parameters", {}),
                        "optimizable": bool(getattr(step, "optimizable", False)),
                        "interchangeable": bool(getattr(step, "is_interchangeable", False)),
                        "alternatives": [{
                            "alias": getattr(alternative, "_flow_alias", None),
                            "variant_id": getattr(alternative, "_flow_variant_id", None),
                            "component": f"{type(alternative).__module__}.{type(alternative).__qualname__}",
                        } for alternative in getattr(step, "_flow_alternatives", ())],
                    }) if getattr(step, "_flow_explicit", False) else None,
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
            "metric_coverage": deepcopy(self.metric_coverage),
            "metric_report": deepcopy(self.metric_report),
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
        self.computed_metrics = {}
        self.fold_metrics = []
        self.training_audit = None
        self.metric_coverage = {}
        self.metric_report = []
        if not self.pipeline.have_model:
            return None
        metrics: list[dict] = []
        fold_metrics: list[dict[str, Any]] = []

        splitter_fingerprint = (
            hash_evaluation_context(splitter) if cache_split else None
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
                # Fit in two steps to allow caching.
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
                        aggregated_metrics={},
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
                    fold_number=fold_index,
                    model_only=True)
                if not self.__valid_main_metric(fold_result):
                    raise ValueError(
                        f"Main metric '{self.main_metric}' is missing or invalid on fold {fold_index}"
                    )
                metrics.append(fold_result)
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
                    f"Skip candidate {self._pipeline_signature()} after metric failure: {exc!r}"
                )
                if store_audit:
                    self.training_audit = self.build_training_audit(
                        dataset=dataset,
                        fold_metrics=fold_metrics,
                        aggregated_metrics={},
                        status="failed",
                        error=f"metric failure: {exc!r}",
                    )
                return {}

        if cache_key and not from_cache:
            Cache().add_to_cache(cache_key, dataset_key, to_cache)

        computed_metrics = self.__aggregate_metrics(metrics) if metrics else {}
        if not self.__valid_main_metric(computed_metrics):
            if store_audit:
                self.training_audit = self.build_training_audit(
                    dataset, fold_metrics, {}, status="failed",
                    error=f"Main metric '{self.main_metric}' has no valid aggregate",
                )
            return {}
        self.computed_metrics = computed_metrics
        self.fold_metrics = deepcopy(fold_metrics)
        if store_audit:
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

        training_report = self.metric_report
        self.metric_report = []
        try:
            return self.__compute_metrics(X, np.array(y))
        finally:
            self.evaluation_report = self.metric_report
            self.metric_report = training_report

    def __compute_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        pipeline : IAMLPipeline = None,
        X_train: pd.DataFrame = None,
        y_train: np.ndarray = None,
        fold_number: int | None = None,
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
        metric_items = self._metric_items()
        needs = {metric.needed_prediction for _, metric in metric_items}
        context = {"fold": fold_number} if fold_number is not None else {}

        for need in needs:
            try:
                method = getattr(pipeline, need)
                y_pred = method(X_test, **kwargs)
                for key, metric in metric_items:
                    try:
                        if metric.needed_prediction == need:
                            value = metric.compute(
                                y_test,
                                y_pred,
                                y_train=y_train,
                                X_train=X_train
                            )
                            if not self.__finite_metric(value):
                                raise ValueError("Metric output must be a finite numeric scalar")
                            computed[key] = value
                            self.metric_report.append({"key": key, "status": "success", **context})
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        Logger().warning(f"Metric '{key}' failed: {exc!r}")
                        self.metric_report.append({"key": key, "status": "error", "reason": repr(exc), **context})
            except Exception as exc:
                status = "inapplicable" if isinstance(exc, AttributeError) else "error"
                Logger().warning(f"Pipeline {self._pipeline_signature()} cannot produce '{need}': {exc!r}")
                self.metric_report.extend(
                    {"key": key, "status": status, "reason": repr(exc), **context}
                    for key, metric in metric_items if metric.needed_prediction == need
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
        rows = []
        for key, metric in self._metric_items():
            value = self.computed_metrics.get(key)
            formatted = f"{value:.4f}" if isinstance(value, Real) else "unavailable"
            rows.append(f'| `{key}` | **{formatted}** | *{metric.explain()}* |')
        metrics = '\n            '.join(rows)

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
        """Run all pipeline steps to predict class probabilities.

        :param pd.DataFrame X: Features used as candidate of the pipeline.
        :return: Class probabilities in the predictor's class order.
        :raise AttributeError: The model does not support probability predictions.
        """
        return self.pipeline.predict_proba(X)
