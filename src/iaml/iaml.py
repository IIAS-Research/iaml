"""Integrated AutoML for Medical Labs (IAML).

Search and evaluate modular prediction pipelines for clinical research with
tabular data. Trained candidates expose their pipeline steps, evaluation metrics
and explanation methods for inspection and study reporting.
"""
from copy import deepcopy
import time
import math
import multiprocessing
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from .timed_pool_executor import TimedPoolExecutor, TerminatedError
from .step import Step
from .cache import Cache
from .cache_keys import hash_evaluation_context
from .metastep import MetaStep
from .candidate import Candidate
from .dataset import Dataset
from .metric import Metric
from .statistic import Statistic
from .worker_manager import WorkerManager
from .splitters import kfold_splitter
from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep
from .meta_partial_explorer_step import MetaPartialExplorerStep
from .optimizers import Optimizer, GeneticOptimizer, RandomOptimizer, BayesianOptimizer
from .predictor import Predictor
from .logger import Logger
from .plot import StatisticPlot
from .actionables.cleaning.act_simple_imputer import ActSimpleImputer
from .actionables.normalize.act_standard_scaler import ActStandardScaler
from .sklearn_preprocessor import SklearnPreprocessor

# Default Actionables -> Must be a wildcard import to help IAML to know all available the steps
from .actionables import * # pylint: disable=unused-wildcard-import,wildcard-import

# Default Wrappers -> Must be a wildcard import to help IAML to know all available the steps
from .wrapper import * # pylint: disable=unused-wildcard-import,wildcard-import

# cuDF pandas acceleration
try:
    import cudf.pandas
    cudf.pandas.install()
    Logger().info('cuDF is installed: using cuDF pandas accelerator mode.')
except ImportError as e:
    Logger().warning('cuDF not found: falling back to standalone pandas.')

if TYPE_CHECKING:
    from .iaml_pipeline import IAMLPipeline


class IAML:  # pylint: disable=too-many-instance-attributes
    """Configure and search prediction pipelines for a clinical research dataset.

    :meth:`fit` returns trained candidates for evaluation, pipeline inspection
    and explanation of predictions.

    :param int, optional max_workers: Maximum parallel workers. Default to cpu count.
    :param int, optional max_stage_duration: Maximum duration of a stage. Default to None.
    :param callable, optional splitter: Split function to use. Default to kfold_splitter.
    :param int, optional max_duration: Search time budget. -1 means no global limit.
    :param int | str, optional time_before_sample_use: Time before we use sampled data. 
        Default to None.
    :param bool, optional preprocessor: Use preprocessor. Default to False.
    :param main_metric: Metric instance preserving its parameters, or the result key
        of a configured metric. None selects the default objective for the task.
    :param Optimizer, optional optimizer: Optimizer class to use. Default to GeneticOptimizer.
    :param int, optional train_on_n_samples: Limit the initial search dataset to this many
        rows. None or nonpositive values use all rows.
    :param bool, optional keep_training_history: If True, store detailed CV audit records for
        every evaluated pipeline. Default to False.
    :param bool, optional refit_on_sample: Reuse the initial train_on_n_samples sample for
        final fitting. If False, refit on all input rows. Defaults to True. Has no effect
        without a positive train_on_n_samples limit.
    :param initial_preprocessor: Optional clonable sklearn transformer. It must return
        a numeric DataFrame with unchanged rows and index. Every generated pipeline,
        including minimalist candidates, starts with this mandatory transformer.
        It is fitted afresh within each CV training fold and during final fitting.
    :param pipeline: Declarative training recipe. None uses the shared preset with
        visible ``minimal`` and ``main`` branches. Supplied recipes are copied.
    :param metrics: Metric collection or list. None uses the default family.
        Aliases identify scores and may select the main objective.
    :param statistics: Descriptive collection or list. None uses the default family;
        an empty list disables collective descriptive calculations.
    :param explanations: Explanation collection or list. None uses the default family;
        an empty list disables collective explanations. Calculations remain on demand.
    """
    def __init__( # pylint: disable=too-many-arguments
        self,
        max_workers: int = None,
        max_stage_duration: int = None,
        splitter: callable = None,
        max_duration: int = -1,
        time_before_sample_use: int | str = None,
        preprocessor: bool = False,
        main_metric: Metric | str = None,
        optimizer: Optimizer = GeneticOptimizer,
        train_on_n_samples: int = None,
        keep_training_history: bool = False,
        refit_on_sample: bool = True,
        initial_preprocessor: Any = None,
        *,
        pipeline=None,
        metrics=None,
        statistics=None,
        explanations=None) -> None:
        # Set pandas config to avoid SettingsWithcopyWarning
        pd.options.mode.copy_on_write = True

        self.preprocessor: bool = preprocessor
        """Enable / Disable preprocessor"""

        self.optimizer = optimizer
        """Choose Optimizer"""
        
        self.train_on_n_samples = train_on_n_samples
        """If defined, pick n sample in the dataset before train"""

        self.refit_on_sample: bool = refit_on_sample
        """Apply the explicit search sample limit to final fitting as well."""

        self.initial_preprocessor = initial_preprocessor
        """Unfitted transformer template, prepended to all candidate pipelines."""

        self.keep_training_history: bool = keep_training_history
        """Whether to store detailed cross-validation audit records."""

        self.training_history: list[dict[str, Any]] = []
        """Detailed audit records for evaluated pipelines during the last fit."""

        self._training_history_seen: set[tuple[Any, ...]] = set()
        """Deduplicate audit records across warmup, cache hits, and repeated evaluations."""

        # Set max duration of each stage
        if max_stage_duration is None:
            self.max_stage_duration = max(max_duration / 5, 900)
            Logger().warning(
                f"Max duration of each stage was set to {self.max_stage_duration} seconds")
        else:
            self.max_stage_duration = max_stage_duration

        self.splitter: callable = splitter if splitter is not None else kfold_splitter
        """Splitter callable"""

        self.main_metric: Metric | str = main_metric
        """Main metric"""

        self.max_duration: int = max_duration
        """Maximum training duration"""

        if time_before_sample_use == 'auto' and max_duration:
            self.time_before_sample_use = max(max_duration / 5, 60)
        elif time_before_sample_use:
            self.time_before_sample_use = time_before_sample_use
        else:
            self.time_before_sample_use = math.inf

        self.candidates: list[Candidate] = None
        """list of Candidates for this training"""

        self.init_candidate: Candidate = None
        """Initial candidate"""

        self.first_step: Step = None # Will be the first Step of the pipeline (probably a MetaStep
        """Hold the first step of the pipeline"""

        self.last_stage_candidates: list[Candidate] = []
        """Hold last generated candidates"""

        self.executor: TimedPoolExecutor = None
        """Hold TimePoolExecutor"""

        self._pipeline_spec = None
        self._flow_execution_root = None
        self._flow_compiled_first_step = None
        self.minimal_predictor_step = None
        self._flow_active = False
        self._study_snapshot = None
        self._analysis_reports = {}
        self._descriptive_cache = {}
        self._metrics_explicit = metrics is not None
        from .flow import metrics as metric_family, statistics as statistic_family
        from .flow import explanations as explanation_family
        self.metrics = (metric_family() if metrics is None else
                        metrics.clone() if hasattr(metrics, 'clone') else metrics)
        self.statistics = (statistic_family() if statistics is None else
                           statistics.clone() if hasattr(statistics, 'clone') else statistics)
        self.explanations = (explanation_family() if explanations is None else
                             explanations.clone() if hasattr(explanations, 'clone') else explanations)
        self._metrics_explicit = metrics is not None
        if pipeline is None:
            self.default_pipeline()
        else:
            from .flow.model import Recipe
            if not isinstance(pipeline, Recipe) or pipeline.kind != 'pipeline':
                raise TypeError('pipeline must be an IAML training recipe')
            self.pipeline = pipeline.clone()
            from .flow.compiler import RecipeValidationError
            try:
                self._install_flow_pipeline()
            except RecipeValidationError:
                # An incomplete attached recipe remains editable until launch.
                pass
        self.max_workers = max_workers if (max_workers is not None and max_workers > 0) \
            else multiprocessing.cpu_count()
        """Hold maximum number of parallel workers"""

        self.chosen_candidate: Candidate = None
        """Hold the best candidate"""
        WorkerManager(max_workers=self.max_workers)

        self.descriptive_statistics: pd.DataFrame | None = None
        """Cached descriptive statistics for the last fitted dataset."""

        self._last_dataset: Dataset | None = None
        """Dataset used for the most recent fit, for on-demand statistics."""

    def __del__(self):
        """Delete the TimedPoolExecutor"""
        del self.executor

    def load_pipeline(self, pipeline: dict) -> None:
        """Load any kind of pipeline

        :param dict pipeline: JSON description of the pipeline
        """
        self.first_step = Step.from_pipeline(pipeline)
        self.minimal_predictor_step = None
        self._pipeline_spec = None
        self._flow_execution_root = None
        self._flow_active = False

    @property
    def pipeline(self):
        """Editable declarative recipe belonging to this study."""
        return self._pipeline_spec

    @pipeline.setter
    def pipeline(self, recipe):
        from .flow import PipelineSpec
        if not isinstance(recipe, PipelineSpec):
            # A composed fragment is also a valid complete recipe.
            from .flow.model import Recipe
            if not isinstance(recipe, Recipe):
                raise TypeError('pipeline must be an IAML recipe')
        if recipe.kind != 'pipeline':
            raise TypeError('pipeline requires a training recipe')
        if getattr(recipe, '_owner', None) not in (None, self):
            recipe = recipe.clone()
        previous = getattr(self, '_pipeline_spec', None)
        if previous is not None and previous is not recipe:
            previous._owner = previous._owner_field = None
        self._pipeline_spec = recipe.attach(self, 'pipeline')

    def _set_analysis_collection(self, field, value):
        from .flow import metrics, statistics, explanations
        from .flow.model import AnalysisCollection
        factory = {'metrics': metrics, 'statistics': statistics,
                   'explanations': explanations}[field]
        if not isinstance(value, AnalysisCollection):
            if not isinstance(value, (list, tuple)):
                raise TypeError(f'{field} must be a {field} collection or a list')
            value = factory(*value) if value else AnalysisCollection(field, [])
        if value.kind != field:
            raise TypeError(f'{field} requires a collection of the same kind')
        if getattr(value, '_owner', None) not in (None, self):
            value = value.clone()
        previous = getattr(self, '_' + field + '_spec', None)
        if previous is not None and previous is not value:
            previous._owner = previous._owner_field = None
        return value.attach(self, field)

    @property
    def metrics(self):
        return self._metrics_spec

    @metrics.setter
    def metrics(self, value):
        self._metrics_spec = self._set_analysis_collection('metrics', value)
        self._metrics_explicit = True

    @property
    def statistics(self):
        return self._statistics_spec

    @statistics.setter
    def statistics(self, value):
        self._statistics_spec = self._set_analysis_collection('statistics', value)

    @property
    def explanations(self):
        return self._explanations_spec

    @explanations.setter
    def explanations(self, value):
        self._explanations_spec = self._set_analysis_collection('explanations', value)

    def _install_flow_pipeline(self):
        from .flow.compiler import compile_pipeline
        root = compile_pipeline(self.pipeline, optimizer=self.optimizer,
                                preprocessor=self.preprocessor,
                                fast=getattr(self, '_flow_fast', False))
        self._flow_execution_root = root
        # Keep the principal execution tree available to historical low-level users.
        main = next((step for step in getattr(root, 'steps', [])
                     if getattr(step, '_flow_alias', None) == 'main'), None)
        self.first_step = main if main is not None else root
        self._flow_compiled_first_step = self.first_step
        self.minimal_predictor_step = None
        self._flow_active = True

    def _prepare_flow(self):
        if getattr(self, '_pipeline_spec', None) is None:
            return False
        if (self.first_step is not getattr(self, '_flow_compiled_first_step', None)
                or self.minimal_predictor_step is not None):
            # Explicit legacy execution-tree overrides remain supported.
            self._flow_active = False
            return False
        self._install_flow_pipeline()
        return True


    def default_pipeline(self, fast: bool = False) -> None:
        """Install the shared default recipe, including its visible minimal branch."""
        from .flow import PipelineSpec
        self._flow_fast = fast
        self.pipeline = PipelineSpec.default()
        self._install_flow_pipeline()

    def __callback(self, callback: callable, **kwargs: dict) -> None:
        """Call callback function if defined
        
        :param callable callback: Function to call.
        :param dict, optional \\**kwargs: Additional parameters.
        """
        if callback and callable(callback):
            callback(**kwargs)

    def baseline( # pylint: disable=too-many-arguments
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: pd.DataFrame = None,
        groups_columns: list[str] = None,
        generation_sample_size: int = 200,
        verbose: int = 1) -> Candidate:
        """Run a very basic pipeline to train a model baseline 
        
        :param pd.DataFrame X: Training features 
        :param pd.DataFrame y: Training labels
        :param pd.DataFrame, optional groups: Dataframe used to split data by groups. 
            Default to None.
        :param list[str], optional groups_columns: List of column names used to split data by 
            groups. Default to None.
        :param int, optional generation_sample_size: Size of the sample dataset used to generate 
            first generation of candidates (default 200).
        :param int, optional verbose: Verbosity level. Default to 1.
        :return: Baseline candidate
        """
        # Avoid [] dangerous default value in the signature
        if groups_columns is None:
            groups_columns = []

        # Create a baseline pipeline
        baseline_pipe = MetaOrderedStep(tag="Main") # First step -> Contain all pipeline's stages
        baseline_pipe.add_step(MetaStep(tag='baseline_cleaning'))
        baseline_pipe.add_step(MetaExplorerStep(tag='baseline_predictor'))

        Logger().verbose = verbose # Set logger verbose

        dataset:Dataset = Dataset(
            deepcopy(X),
            deepcopy(y),
            groups=groups,
            groups_columns=groups_columns)

        ### INITIAL GENERATE CANDIDATE
        init_candidate: Candidate = Candidate(
            dataset.sample(generation_sample_size),
            main_metric=self.main_metric)

        # Select metrics used to evaluate performances
        for metric \
            in self.__metrics_selection(dataset.X, dataset.y, dataset.type_of_target):
            init_candidate.add_metric(metric)

        # Generate candidates
        candidates = baseline_pipe.run(init_candidate)

        # Remove candidate without predictor
        candidates = [candidate for candidate in candidates \
            if candidate.pipeline.predictor is not None]

        for candidate in candidates:
            candidate.pipeline.fit(dataset.X, dataset.y)

        return candidates

    ##################
    ### PROPERTIES ###
    ##################

    # Dataset from candidate data
    @property
    def dataset(self) -> Dataset:
        """Shortcut to get candidate Dataset

        :return: Candidate dataset defined by .fit()
        """
        return self.init_candidate.dataset

    ###########
    ### RUN ###
    ###########

    def fit( # pylint: disable=too-many-arguments,too-many-locals
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: pd.DataFrame = None,
        groups_columns: list[str] = None,
        patience: int = -1,
        generation_sample_size: int = 200,
        n_candidates: int = 1,
        callback: callable = None,
        verbose: int = 1,
        log_callback: callable = None) -> list[Candidate]:
        """Run Pipeline to fit steps and models on X & y data. 

        :param pd.DataFrame X: Training features 
        :param pd.DataFrame y: Training labels
        :param pd.DataFrame, optional groups: Dataframe used to split data by groups. 
            Default to None.
        :param list[str], optional groups_columns: List of column names used to split data by 
            groups. Default to None.
        :param int, optional patience: Max generation without improvement. Default to -1.
        :param int, optional generation_sample_size: Size of the sample dataset used to generate 
            first generation of candidates (default 200).
        :param int, optional n_candidates: Number of candidates to return. Default to 1.
        :param callable, optional callback: Method call after each big step of training.
        :param int, optional verbose: Verbosity level. Default to 1.
        :param callable, optional log_callback: Callback for logger.
        :return: List of all the generated candidates. Sorted by performances.
        """
        # Avoid [] dangerous default value in the signature
        if groups_columns is None:
            groups_columns = []

        self._prepare_flow()
        self.check_pipeline() # Raise error if the pipeline is not valid

        Logger().verbose = verbose # Set logger verbose
        if log_callback is not None:
            Logger().set_callback(log_callback)

        start_time = time.monotonic()
        self.executor = TimedPoolExecutor(max_workers=self.max_workers)
        self.training_history = []
        self._training_history_seen = set()

        def remain_time():
            if self.max_duration == -1:
                return math.inf
            return max(0.0, self.max_duration - (time.monotonic() - start_time))

        try:
            refit_X, refit_y, refit_groups_columns = X, y, groups_columns
            dataset:Dataset = Dataset(
                deepcopy(X),
                deepcopy(y),
                groups=groups,
                groups_columns=groups_columns)

            if self.train_on_n_samples != None and self.train_on_n_samples > 0:
                dataset = dataset.sample(self.train_on_n_samples)
                if self.refit_on_sample:
                    # Preserve this sample even if the search later downsizes again.
                    # Dataset.X already excludes the grouping columns.
                    refit_X, refit_y, refit_groups_columns = dataset.X, dataset.y, []

            self._last_dataset = dataset
            self.descriptive_statistics = None

            ### INITIAL GENERATE CANDIDATE
            metric_definitions = None
            explanation_definitions = None
            statistic_definitions = None
            objective = self.main_metric
            self._study_snapshot = None
            analysis_specs = {}
            if hasattr(self, '_explanations_spec'):
                from .study_analyses import compile_analyses
                analysis_specs = {field: getattr(self, field).clone().freeze()
                                  for field in ('metrics', 'statistics', 'explanations')}
                explanation_definitions = compile_analyses(analysis_specs['explanations'])
                # Validate result keys before fitting, even for on-demand analyses.
                statistic_definitions = compile_analyses(analysis_specs['statistics'])
            metric_recipe = getattr(self, '_metrics_spec', None)
            metric_recipe_edited = (metric_recipe is not None and (
                metric_recipe.tag is None or metric_recipe.children
                or metric_recipe.excluded or metric_recipe._removed_ids
                or any(getattr(node, '_declared_params', ()) or node.alias
                       for node in metric_recipe._walk() if hasattr(node, 'parameters'))))
            if (metric_recipe is not None
                    and (getattr(self, '_flow_active', False)
                         or getattr(self, '_metrics_explicit', False)
                         or metric_recipe_edited)):
                from .study_analyses import compile_metrics
                metric_definitions, objective, metric_report = compile_metrics(
                    analysis_specs['metrics'], self.main_metric, dataset)
            self.init_candidate: Candidate = Candidate(
                dataset.sample(generation_sample_size),
                main_metric=objective,
                metric_definitions=metric_definitions,
                explanation_definitions=explanation_definitions,
                study_snapshot=self._study_snapshot if hasattr(self, '_study_snapshot') else None)

            if self.initial_preprocessor is not None:
                # Encode the generation sample so both branches can discover
                # suitable predictors. Keep the raw search/refit datasets: the
                # Step is cloned/refitted inside every pipeline's CV fit.
                initial_step = SklearnPreprocessor(self.initial_preprocessor)
                initial_step.fit(self.init_candidate.dataset)
                self.init_candidate = self.init_candidate.add_to_pipeline(initial_step)

            # Select metrics used to evaluate performances
            if metric_definitions is None:
                for metric in self.__metrics_selection(
                        dataset.X, dataset.y, dataset.type_of_target):
                    self.init_candidate.add_metric(metric)

            minimal_candidates = ([] if getattr(self, '_flow_active', False) else
                                  self.__generate_minimal_candidates(self.init_candidate))

            # Generate candidates
            pipeline_candidates = self.__run(self.init_candidate)
            pipeline_candidates = [candidate for candidate in pipeline_candidates \
                if candidate.pipeline.predictor is not None]

            candidates = minimal_candidates + pipeline_candidates
            # Remove candidate without predictor
            candidates = [candidate for candidate in candidates \
                if candidate.pipeline.predictor is not None]
            self.candidates = candidates

            if not candidates and getattr(self, '_flow_active', False):
                raise ValueError('No applicable candidate pipeline matches the declared recipe')

            if minimal_candidates:
                Logger().info(f"{len(minimal_candidates)} minimalist pipelines generated")
            Logger().info(f"{len(candidates)} generated pipelines")


            # Warmup is real CV, so it must use the same interruptible executor
            # and shared search/stage budgets as every subsequent evaluation.
            # Minimal candidates already lead the pool and provide a quick,
            # honestly evaluated starting point without removing full pipelines.
            warmup_candidate = None
            if candidates and remain_time() >= 1:
                # A single slow baseline must leave time to evaluate the other
                # candidates. Unlimited searches retain the stage duration cap.
                remaining = remain_time()
                warmup_timeout = remaining / 5
                Logger().info(
                    f"Warming up (at most {min(warmup_timeout, self.max_stage_duration):.2f}s): "
                    f"{candidates[0].pipeline.name}")
                warmup_candidates = self.__run_evaluations(
                    candidates[:1], dataset, timeout=remaining, callback=callback,
                    stage_timeout=warmup_timeout)
                if warmup_candidates:
                    warmup_candidate = warmup_candidates[0]
                    Logger().info("Warmed up !")
                else:
                    Logger().info("No warmup result within the stage budget.")
                    # Try alternatives before retrying the same candidate,
                    # especially when only one worker is available.
                    candidates = candidates[1:] + candidates[:1]

            ### INITIAL EVALUATION
            # Evaluate candidates
            gen0_candidates = []
            i = 0
            can_be_downsize = True
            while can_be_downsize and not gen0_candidates and remain_time() >= 1:
                # If process is too long and dataset big enough,
                # we can downsize it to get quicker training
                if i > 0:
                    dataset = dataset.sample(0.1)
                    Logger().warning(f"Training is too time consuming. \
                        Let's try again with dataset sample. \
                        New features shape {dataset.X.shape}")
                i+= 1
                can_be_downsize = dataset.X.shape[0] >= 500

                timeout = min(remain_time(), self.time_before_sample_use) \
                    if can_be_downsize else remain_time()

                gen0_candidates = self.__run_evaluations(candidates,
                            dataset, timeout=timeout, callback=callback)

            if not gen0_candidates:
                if warmup_candidate and warmup_candidate.computed_metrics:
                    Logger().warning(
                        "No candidates evaluated before timeout; using warmup candidate."
                    )
                    gen0_candidates = [warmup_candidate]
                elif remain_time() < 1:
                    raise TimeoutError('IAML was unable to generate a model within the \
                        imposed time limit. Try increasing the processing time')
                else:
                    raise RuntimeError('Undefined error. IAML was unable to create pipeline \
                        based on your data')

            ### FINETUNING
            candidates = self.__optimize(dataset,
                                        gen0_candidates,
                                        optimizer=self.__build_optimizer(remain_time()),
                                        max_duration=remain_time(),
                                        patience=patience,
                                        callback=callback)

            ### FINAL FIT
            self.executor.shutdown()

            # Fit candidates on the requested sample or all original input rows.
            fit_candidates = []
            last_fit_error = None
            for candidate in candidates:
                if len(fit_candidates) >= n_candidates:
                    break
                Cache.reset()
                current_candidate = deepcopy(candidate)
                try:
                    Logger().info(f"Final fit: {len(refit_X)} rows, {current_candidate.pipeline.name}")
                    current_candidate.pipeline.fit(
                        refit_X,
                        refit_y,
                        groups_columns=refit_groups_columns,
                        metrics=current_candidate.metrics,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    Logger().warning(
                        f"Skipping candidate during final fit after failure: {exc!r}"
                    )
                    last_fit_error = exc
                    continue
                fit_candidates.append(current_candidate)

            if not fit_candidates:
                details = f" Last error: {last_fit_error!r}" if last_fit_error else ""
                raise ValueError(f"IAML could not fit any candidate pipeline.{details}")

            self.chosen_candidate = fit_candidates[0]
            self.last_stage_candidates = candidates

            return fit_candidates
        except TerminatedError:
            Logger().info('IAML was terminated.')
        except Exception as ex:
            Logger().error('Error during fit')
            raise ex
        finally:
            self.executor.shutdown()

    @property
    def chosen_model(self) -> 'IAMLPipeline':
        """Return the best model trained with fit

        :return: Best predictor pipeline
        """
        if not self.chosen_candidate:
            return None

        return self.chosen_candidate.pipeline

    def visualize_descriptive_statistics(self) -> list[StatisticPlot]:
        """Return a list of plots that show descriptive statistics."""
        if hasattr(self, '_statistics_spec') and self._last_dataset is not None:
            self.get_descriptive_statistics()
        if self.descriptive_statistics is None and self._last_dataset is not None:
            self.__ensure_descriptive_statistics(self._last_dataset)

        if self.descriptive_statistics is None or self.descriptive_statistics.empty:
            return []

        plots: list[StatisticPlot] = []
        from .study_analyses import canonical_statistics
        stats_df = canonical_statistics(self.descriptive_statistics)

        def group_columns_by_feature(dataframe: pd.DataFrame) -> dict[str, list[str]]:
            columns = list(dataframe.columns)
            base_names: list[str] = []
            for col in columns:
                if isinstance(col, str) and col.endswith('_all'):
                    base = col[:-4]
                    if base not in base_names:
                        base_names.append(base)

            groups: dict[str, list[str]] = {}
            used_cols: set[str] = set()
            if base_names:
                for base in base_names:
                    group = [
                        col for col in columns
                        if col == base or (isinstance(col, str) and col.startswith(f"{base}_"))
                    ]
                    groups[base] = group
                    used_cols.update(group)

            for col in columns:
                if col not in used_cols:
                    groups[col] = [col]

            return groups

        for plot_sub_class in StatisticPlot.__subclasses__():
            if not getattr(plot_sub_class, 'enabled', True):
                continue
            if getattr(plot_sub_class, 'group_by_feature', False):
                for base, cols in group_columns_by_feature(stats_df).items():
                    plot = plot_sub_class().compute(stats_df[cols], base_name=base)
                    plots.append(plot)
            else:
                plots.append(plot_sub_class().compute(stats_df))

        return plots

    def get_descriptive_statistics(self, X=None, y=None) -> pd.DataFrame:
        """Describe supplied data, or the last raw search dataset, without fitting."""
        if X is None and y is not None:
            raise ValueError('X and y must be supplied together')
        if X is not None:
            if y is None:
                raise ValueError('The explicit descriptive request requires y')
            dataset = Dataset(deepcopy(X), deepcopy(y))
        else:
            dataset = self._last_dataset
        if dataset is None:
            return pd.DataFrame()
        if hasattr(self, '_statistics_spec'):
            from .study_analyses import (compile_analyses, compute_statistics,
                                         analyses_signature, copy_statistics)
            definitions = compile_analyses(self.statistics)
            signature = analyses_signature(definitions)
            key = (dataset.fingerprint(), signature)
            if signature is not None and key in self._descriptive_cache:
                table, report = self._descriptive_cache[key]
            else:
                table, report = compute_statistics(definitions, dataset)
                if signature is not None:
                    self._descriptive_cache[key] = (copy_statistics(table), deepcopy(report))
            self._analysis_reports['statistics'] = deepcopy(report)
            if X is None:
                self.descriptive_statistics = copy_statistics(table)
            return copy_statistics(table)
        if X is not None:
            return self.__compute_descriptive_statistics(dataset).copy(deep=True)
        self.__ensure_descriptive_statistics(dataset)
        return (self.descriptive_statistics.copy(deep=True)
                if self.descriptive_statistics is not None else pd.DataFrame())

    def __ensure_descriptive_statistics(self, dataset: Dataset) -> None:
        """Compute descriptive statistics once, for on-demand usage."""
        if hasattr(self, '_statistics_spec'):
            self.get_descriptive_statistics()
            return
        if self.descriptive_statistics is not None:
            return

        try:
            self.descriptive_statistics = self.__compute_descriptive_statistics(dataset)
        except Exception as exc:  # noqa: BLE001
            Logger().warning(f"Descriptive statistics computation failed: {exc}")
            self.descriptive_statistics = pd.DataFrame()

    def __compute_descriptive_statistics(self, dataset: Dataset) -> pd.DataFrame:
        """Compute descriptive statistics on the dataset."""
        computed_statistics = pd.DataFrame()
        for statistic_sub_class in Statistic.all_subclasses():
            statistic = statistic_sub_class()
            if statistic.suitable(dataset):
                result = statistic.compute(dataset)
                if result is not None and not result.empty:
                    computed_statistics = pd.concat([computed_statistics, result])
        return computed_statistics

    def check_pipeline(self) -> None:
        """Raise Exception if pipeline is not valid
        
        :raise AttributeError: Step Pipeline must contains at least one predictor
        """
        steps = self.__all_steps()

        # Pipeline must have at least one predictor
        if not any((Predictor in s.__class__.__mro__) for s in steps if s.enable):
            raise AttributeError('Step Pipeline must contains at least one predictor')

        # TODO Others tests ?

    def __run_evaluations(
        self,
        candidates: list[Candidate],
        dataset: Dataset,
        timeout: float = None,
        stage_number: int = None,
        callback: callable = None,
        stage_timeout: float = None) -> list[Candidate]:
        """Evaluate candidates
        
        :param list[Candidate] candidates: Candidates to evaluate.
        :param Dataset dataset: Dataset used for evaluation.
        :param float, optional timeout: Budget including preparation and submission.
            Defaults to the stage duration limit.
        :param int, optional stage_number: Stage number running. Default to None.
        :param callable, optional callback: Method called after evaluation. Default to None.
        :param float, optional stage_timeout: Additional cap for this evaluation only;
            the callback still reports the remaining ``timeout`` budget.
        """
        start_time = time.monotonic()
        budget = self.max_stage_duration if timeout is None else min(timeout, self.max_stage_duration)
        if stage_timeout is not None:
            budget = min(budget, stage_timeout)
        deadline = start_time + max(0.0, budget)
        new_candidates: list[Candidate] = []
        splitter_fingerprint = hash_evaluation_context(self.splitter)
        dataset_key = dataset.fingerprint() if splitter_fingerprint is not None else None
        evaluation_cache_keys: set[str] = set()
        with Logger().progress as progress:
            task = progress.add_task(
                f'Stage {stage_number}' if stage_number is not None else "Initial evaluation",
                total=len(candidates))

            def update_progressbar(*args): # pylint: disable=unused-argument
                progress.update(task, advance=1)

            self.executor.set_callback(update_progressbar)
            for candidate in candidates:
                if time.monotonic() >= deadline:
                    break
                cache_key = self.__evaluation_cache_key(candidate, splitter_fingerprint)
                if cache_key is not None:
                    evaluation_cache_keys.add(cache_key)
                from_cache = Cache().from_cache(cache_key, dataset_key) if cache_key else None

                if from_cache:
                    self.__hydrate_cached_candidate(candidate, from_cache, dataset)
                    new_candidates.append(candidate)
                    update_progressbar() # Update progressbar even if data come from cache
                else:
                    submitted = self.executor.submit(
                            process_executor,
                            candidate,
                            dataset,
                            deadline=deadline,
                            splitter=self.splitter,
                            store_audit=self.keep_training_history,
                        )
                    if not submitted:
                        break

            # Preparation and submission have already consumed part of the budget.
            new_candidates += self.executor.join(max(0.0, deadline - time.monotonic()))
            self.__collect_training_history(new_candidates)

            if new_candidates:
                skipped = sum(1 for candidate in new_candidates if not candidate.computed_metrics)
                if skipped:
                    Logger().warning(
                        f"Skipped {skipped} candidates with no computed metrics."
                    )
                new_candidates = [candidate for candidate in new_candidates if candidate.computed_metrics]

            new_candidates.sort(reverse=True)

            # Add results to progressbar
            if new_candidates:
                progress.tasks[task].description = f'{progress.tasks[task].description} \
                    ({new_candidates[0].get_main_metric_value():.4f})'
            else:
                progress.tasks[task].description = f'{progress.tasks[task].description} \
                    (no result)'

        # Add to cache
        for candidate in new_candidates:
            cache_key = self.__evaluation_cache_key(candidate, splitter_fingerprint)
            if cache_key in evaluation_cache_keys and not Cache().from_cache(cache_key, dataset_key):
                Cache().add_to_cache(
                    cache_key,
                    dataset_key,
                    self.__build_cached_candidate(candidate),
                )

        best_metric = new_candidates[0].get_main_metric_value() if new_candidates else None
        remaining_time = (budget if timeout is None else timeout) - (time.monotonic() - start_time)

        self.__callback(callback, # pylint: disable=too-many-function-args
            generation = stage_number,
            generation_size = len(new_candidates),
            best = best_metric,
            remaining_time = remaining_time,
            text = f'Stage {stage_number} finished' \
                if stage_number is not None else "Initial evaluation finished")

        return new_candidates

    def __evaluation_cache_key(
        self, candidate: Candidate, splitter_fingerprint: str | None
    ) -> str | None:
        """Keep scores separate for each splitter and metric configuration."""
        if splitter_fingerprint is None:
            return None
        signature = candidate.evaluation_context_signature()
        if signature is None:
            return None
        context = hash_evaluation_context(
            candidate.pipeline.fingerprint(), signature,
            self.keep_training_history,
        )
        if context is None:
            return None
        return f"IAML_{splitter_fingerprint}_{context}"

    def __build_optimizer(self, duration: float) -> Optimizer:
        """Instantiate optimizer."""
        return self.optimizer(duration=None if math.isinf(duration) else duration)

    def __build_cached_candidate(self, candidate: Candidate) -> dict[str, Any] | dict[str, float]:
        """Build the payload stored in cache for evaluated candidates."""
        if (self.keep_training_history and candidate.training_audit is not None
                or getattr(candidate, 'metric_coverage', None)):
            return {
                "__computed_metrics__": deepcopy(candidate.computed_metrics),
                "__training_audit__": (deepcopy(candidate.training_audit)
                                       if self.keep_training_history else None),
                "__metric_coverage__": deepcopy(candidate.metric_coverage),
                "__fold_metrics__": deepcopy(candidate.fold_metrics),
                "__metric_report__": deepcopy(candidate.metric_report),
            }
        return deepcopy(candidate.computed_metrics)

    def __hydrate_cached_candidate(
        self,
        candidate: Candidate,
        payload: dict[str, Any] | dict[str, float],
        dataset: Dataset,
    ) -> None:
        """Restore cached evaluation results into a candidate."""
        candidate.fold_metrics = []
        candidate.training_audit = None

        if isinstance(payload, dict) and "__computed_metrics__" in payload:
            candidate.computed_metrics = deepcopy(payload["__computed_metrics__"])
            candidate.metric_coverage = deepcopy(payload.get('__metric_coverage__', {}))
            candidate.fold_metrics = deepcopy(payload.get('__fold_metrics__', []))
            candidate.metric_report = deepcopy(payload.get('__metric_report__', []))
            audit = payload.get("__training_audit__")
            if audit is not None:
                candidate.training_audit = deepcopy(audit)
                candidate.fold_metrics = deepcopy(audit.get("fold_metrics", []))
                return
        else:
            candidate.computed_metrics = deepcopy(payload)

        if self.keep_training_history:
            candidate.training_audit = candidate.build_training_audit(
                dataset=dataset,
                fold_metrics=[],
                aggregated_metrics=candidate.computed_metrics,
                status="success",
            )

    def __collect_training_history(self, candidates: list[Candidate]) -> None:
        """Collect unique candidate audit records for the last fit."""
        if not self.keep_training_history:
            return

        for candidate in candidates:
            record = getattr(candidate, "training_audit", None)
            if not record:
                continue
            key = (
                record.get("pipeline_fingerprint"),
                record.get("dataset_fingerprint"),
                record.get("status"),
                record.get("error"),
            )
            if key in self._training_history_seen:
                continue
            self._training_history_seen.add(key)
            self.training_history.append(deepcopy(record))

    def __optimize( # pylint: disable=too-many-arguments
        self,
        dataset: Dataset,
        candidates: list[Candidate],
        optimizer: Optimizer = Optimizer(),
        patience: int = 5,
        max_duration: int = -1,
        callback: callable = None) -> list[Candidate]:
        """Optimize candidates.
        
        :param Dataset dataset: Dataset used for optimization.
        :param list[Candidate], optional candidates: Candidates to optimize.
        :param Optimizer, optional optimizer: Optimizer to use.
        :param int, optional patience: Max generation without improvement. Default to 5.
        :param int, optional max_duration: Maximum optimization duration. Default to -1.
        :param callable, optional callback: Method to call after optimization. Default to None.
        :return: list of optimized candidate.
        """
        if not candidates:
            return []

        if max_duration == -1:
            max_duration = math.inf

        # init
        candidates.sort(reverse=True)
        best_result: float = candidates[0].get_main_metric_value()
        best_score: float = candidates[0].get_main_metric_score()
        iterations_without_improvement: int = 0
        iterations_count: int = 0
        duration: int = 0
        starting_time: int = time.monotonic() # seconds

        # If there is not, define an arbitrary stop condition
        if math.isinf(max_duration) and patience == -1:
            Logger().warning('You have not defined any stop condition. \
                Patient has arbitrary set to 20')
            patience = 20

        previous_candidates = candidates

        while   not(optimizer.finished) \
                and (patience == -1 or iterations_without_improvement < patience) \
                and max_duration > duration:
            # Generate new candidates
            generated_candidates = optimizer.run(previous_candidates)

            Logger().info(f'Finetuning... \
                stage={iterations_count} \
                candidates={len(generated_candidates)} \
                patience={iterations_without_improvement}/{patience}, \
                duration={round(duration, 2)}/{max_duration}, \
                best_result={best_result}')

            # Evaluate new candidates
            evaluated_candidates = self.__run_evaluations(generated_candidates,
                        dataset,
                        timeout=max_duration - (time.monotonic() - starting_time),
                        stage_number=iterations_count,
                        callback=callback)

            # Remove not computed (error or timeout)
            evaluated_candidates = [candidate for candidate in evaluated_candidates if candidate.computed_metrics]

            if not evaluated_candidates:
                Logger().warning('No candidates produced a valid evaluation; keeping previous best candidates.')
                candidates = previous_candidates
                break

            candidates = evaluated_candidates
            previous_candidates = candidates

            # Improvement ?
            new_score: float = candidates[0].get_main_metric_score()
            if new_score > best_score:
                best_score = new_score
                best_result = candidates[0].get_main_metric_value()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Duration in seconds
            duration = time.monotonic() - starting_time

            # Increase Iteration count
            iterations_count += 1

        return candidates

    def __metrics_selection(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        type_of_target: str) -> list[Metric]:
        """Select metrics used to evaluate performances

        :param pd.DataFrame X: Training features.
        :param pd.DataFrame y: Training labels
        :param str type_of_target: type of label. Example : continuous, binary
        :return: List of selected metrics
        """
        metrics = []

        for metric_sub_class in Metric.all_subclasses():
            # Reuse the configured main metric instead of resetting its parameters.
            metric = (
                self.main_metric
                if type(self.main_metric) is metric_sub_class
                else metric_sub_class()
            )
            # Verify if a subclass is suitable or not
            if metric is self.main_metric or metric.suitable(X, y, type_of_target):
                metrics.append(metric)
        return metrics

    def __apply_minimal_preprocessing(self, candidate: Candidate) -> Candidate:
        """Ensure minimalist candidates have a basic imputer in their pipeline."""
        dataset = candidate.dataset

        if dataset.X.isna().values.any():
            imputer = ActSimpleImputer()
            results = imputer.run(candidate)

            if isinstance(results, list) and results:
                return results[0]

            return results

        return candidate

    def __generate_minimal_candidates(self, candidate: Candidate) -> list[Candidate]:
        """Generate minimalist candidates using raw predictors only."""
        if not self.minimal_predictor_step:
            return []

        Logger().info('Generate minimalist candidates...')
        minimal_candidate = candidate.to_input()
        minimal_candidate = self.__apply_minimal_preprocessing(minimal_candidate)

        candidates = self.minimal_predictor_step.run(minimal_candidate)

        return [cand for cand in candidates if cand.pipeline.predictor is not None]

    def __run(self, candidate: Candidate) -> list[Candidate]:
        """Run pipeline steps

        :param Candidate candidate: Data used to fit models and steps.
        :return: List of all the generated candidates. Sorted by performances.
        """
        Logger().info("Generate candidate...")
        root = (self._flow_execution_root if getattr(self, '_flow_active', False)
                else self.first_step)
        self.candidates = root.run(candidate)

        return self.candidates

    ########################
    #### CONFIGURATIONS ####
    ########################

    def json_pipeline(self) -> dict:
        """Create a dictionary (JSON) from the loaded Pipeline

        :return: Loaded Pipeline in a JSON format
        """
        if self.first_step is None:
            self._prepare_flow()
        return self.first_step.json_pipeline()

    def all_configurations(self) -> list[dict]:
        """Return a dict with configurations of all steps. 

        :return: Configurations of all steps. 
        """
        if self.first_step is None:
            self._prepare_flow()
        return self.first_step.all_configurations()

    def configure_all(self, configs: dict) -> None:
        """Configure one to many steps with a dict configuration

        :param dict configs: key is a step_id and value is the configuration to set.
        """
        all_steps = self.__all_steps()
        recipe_nodes = ({node.node_id: node for node in self.pipeline._walk()}
                        if self.pipeline is not None else {})

        for step_id, config in configs.items():
            current_step: Step = self.__find_step_by_id(all_steps, step_id)
            if current_step:
                updates = dict(config)
                recipe = recipe_nodes.get(getattr(current_step, '_flow_node_id', None))
                if recipe is not None and hasattr(recipe, 'configure'):
                    recipe.configure(**updates)
                for key, value in updates.items():
                    current_step.configure(key, value)  # pylint: disable=no-member

    def __all_steps(self) -> list[Step]:
        """Recursive method. Return all the pipeline's steps in a list

        :return: All flatten pipelines's steps
        """
        if self.first_step is None:
            self._prepare_flow()
        return self.first_step.all_steps()

    def __find_step_by_id(self, step_list: list[Step], step_id: int) -> Step | None:
        """Find a step by id in a list of step

        :param list[Step] step_list: The step will be searched in this list.
        :param int step_id: Identifier of the step
        :return: Found Step or None
        """
        for step in step_list:
            if id(step) == step_id:
                return step
        return None


def process_executor(candidate: Candidate, *args, **kwargs) -> 'Candidate':
    """Wrap candidate training to run it in subprocess

    :param Candidate candidate: Not trained candidate.
    :param tuple, optional \\*args: Additional parameters.
    :param dict, optional \\**kwargs: Additional parameters.
    :return: Trained candidate.
    """
    # Deepcopy -> Without it, process end is never detected. Strange...
    candidate = deepcopy(candidate)
    candidate.training_evaluate(*args, **kwargs)
    return candidate
