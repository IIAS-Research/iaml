"""
    IAML (Incremental AutoML), a high-performance and modular,
    open-source Python framework. Designed to mimics the behavior
    of a data scientist in creating pipelines and leverages an 
    optimization process inspired by genetic algorithm for
    efficient pipeline construction and hyperparameter tuning.
    
    The framework incorporates explainability features, such as 
    SHAP-based insights, to enhance model transparency and trustworthiness.
"""
from copy import deepcopy
import time
import math
import textwrap
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
    """ Main class of the module.
    IAML will load, configure and fit machine learning pipelines

    :param int, optional max_workers: Maximum parallel workers. Default to cpu count.
    :param int, optional max_stage_duration: Maximum duration of a stage. Default to None.
    :param callable, optional splitter: Split function to use. Default to kfold_splitter.
    :param int, optional max_duration: Search time budget; -1 means no global limit.
    :param int | str, optional time_before_sample_use: Time before we use sampled data. 
        Default to None.
    :param bool, optional preprocessor: Use preprocessor. Default to False.
    :param Metric, optional main_metric: Main metric instance, preserving its parameters.
        Default to None.
    :param Optimizer, optional optimizer: Optimizer class to use. Default to GeneticOptimizer.
    :param int, optional train_on_n_samples: Limit the initial search dataset to this many
        rows. None or nonpositive values use all rows.
    :param bool, optional keep_training_history: If True, store detailed CV audit records for
        every evaluated pipeline. Default to False.
    :param bool, optional refit_on_sample: Reuse the initial train_on_n_samples sample for
        final fitting. If False, refit on all input rows. Default to True; has no effect
        without a positive train_on_n_samples limit.
    """
    def __init__( # pylint: disable=too-many-arguments
        self,
        max_workers: int = None,
        max_stage_duration: int = None,
        splitter: callable = None,
        max_duration: int = -1,
        time_before_sample_use: int | str = None,
        preprocessor: bool = False,
        main_metric: Metric = None,
        optimizer: Optimizer = GeneticOptimizer,
        train_on_n_samples: int = None,
        keep_training_history: bool = False,
        refit_on_sample: bool = True) -> None:
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

        self.main_metric: Metric = main_metric
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

        self.default_pipeline() # Load default pipeline
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
        self.minimal_predictor_step = self.__build_minimal_predictor_step()

    def default_pipeline(self, fast: bool = False) -> None:
        """Load the default pipeline.
        Default pipeline is the recommended way to create classifier and regressor

        Genetic search starts with one normalization and no resampling, then
        explores alternatives through mutations. Other optimizers retain full
        initial exploration because they only change hyperparameters.

        :param bool, optional fast: If true, will only load fast machine learning model.
            Fast mode is use to create fast pipeline and iterate
            quickly when debugging code. Defaults to False.
        """
        self.first_step = MetaOrderedStep(tag="Main") # First step -> Contain all pipeline's stages

        self.first_step.add_step(MetaStep(tag='features_precleaning',
            name='Features Precleaning',
            description=textwrap.dedent('''\
                Converts complex columns into several columns, which helps the
                model to extract information from your data.''')))
        self.first_step.add_step(MetaStep(tag='cleaning',
            name='Features Cleaning',
            description=textwrap.dedent('''\
                Improve data quality, handle missing values, extract
                information from textual columns, etc.''')))
        self.first_step.add_step(MetaStep(tag='features_selection',
            name='Features Selection',
            description=textwrap.dedent('''\
                Decrease number of column to improve the models' performance.''')))
        partial_exploration = (isinstance(self.optimizer, type)
                               and issubclass(self.optimizer, GeneticOptimizer))
        explorer = MetaPartialExplorerStep if partial_exploration else MetaExplorerStep
        normalization_options = {'initial_step': ActStandardScaler()} if partial_exploration else {}
        imbalance_options = {} if partial_exploration else {'also_explore_without': True}
        self.first_step.add_step(explorer(tag='normalize',
            **normalization_options,
            name='Features Normalization',
            description=textwrap.dedent('''\
                Normalize data to help model to give the same interest to each column''')))
        self.first_step.add_step(explorer(tag='imbalance',
            **imbalance_options,
            name='Handle Imbalanced Data',
            description=textwrap.dedent('''\
                Balance the dataset to ensure the model does not favor the
                majority class over the minority class''')))

        if self.preprocessor:
            self.first_step.add_step(
                MetaExplorerStep(tag='features_preprocessing', also_explore_without=True)
            )
        else:
            self.first_step.add_step(
                MetaPartialExplorerStep(
                    tag='features_preprocessing',
                    name="Dimensionality Reduction (optional)",
                    description=textwrap.dedent('''\
                        Reduce the complexity of data and make computations
                        more efficient'''))
            )

        learning_tag = 'fast_predictor' if fast else 'predictor'

        self.first_step.add_step(
            MetaExplorerStep(
                tag=learning_tag,
                name="Machine learning models",
                description="List of machine learning models IAML will try to optimize"))

        self.minimal_predictor_step = self.__build_minimal_predictor_step()

    def __build_minimal_predictor_step(self) -> MetaExplorerStep | None:
        """Build the minimalist predictor stage if suitable models exist."""
        minimal_step = MetaExplorerStep(
            tag='minimal_predictor',
            name='Minimalist Predictors',
            description=textwrap.dedent('''\
                Try high-performing boosting-style models without any preprocessing
                to provide quick baseline candidates before the full pipeline is explored.'''))

        if not minimal_step.steps:
            return None

        return minimal_step

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
            self.init_candidate: Candidate = Candidate(
                dataset.sample(generation_sample_size),
                main_metric=self.main_metric)

            # Select metrics used to evaluate performances
            for metric \
                in self.__metrics_selection(dataset.X, dataset.y, dataset.type_of_target):
                self.init_candidate.add_metric(metric)

            minimal_candidates = self.__generate_minimal_candidates(self.init_candidate)

            # Generate candidates
            pipeline_candidates = self.__run(self.init_candidate)
            pipeline_candidates = [candidate for candidate in pipeline_candidates \
                if candidate.pipeline.predictor is not None]

            candidates = minimal_candidates + pipeline_candidates
            # Remove candidate without predictor
            candidates = [candidate for candidate in candidates \
                if candidate.pipeline.predictor is not None]
            self.candidates = candidates

            if minimal_candidates:
                Logger().info(f"{len(minimal_candidates)} minimalist pipelines generated")
            Logger().info(f"{len(candidates)} generated pipelines")


            Logger().info(f"Warming up...")
            warmup_candidate = pipeline_candidates[0] if pipeline_candidates else candidates[0]
            warmup_candidate.training_evaluate(
                dataset,
                splitter=self.splitter,
                cache_split=False,
                store_audit=self.keep_training_history)
            self.__collect_training_history([warmup_candidate])
            Logger().info(f"Warmed up !")

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
        if self.descriptive_statistics is None and self._last_dataset is not None:
            self.__ensure_descriptive_statistics(self._last_dataset)

        if self.descriptive_statistics is None or self.descriptive_statistics.empty:
            return []

        plots: list[StatisticPlot] = []
        stats_df = self.descriptive_statistics

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

    def get_descriptive_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics, computing them on demand if needed."""
        if self.descriptive_statistics is None and self._last_dataset is not None:
            self.__ensure_descriptive_statistics(self._last_dataset)

        return self.descriptive_statistics if self.descriptive_statistics is not None else pd.DataFrame()

    def __ensure_descriptive_statistics(self, dataset: Dataset) -> None:
        """Compute descriptive statistics once, for on-demand usage."""
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
        callback: callable = None) -> list[Candidate]:
        """Evaluate candidates
        
        :param list[Candidate] candidates: Candidates to evaluate.
        :param Dataset dataset: Dataset used for evaluation.
        :param float, optional timeout: Budget including preparation and submission.
            Defaults to the stage duration limit.
        :param int, optional stage_number: Stage number running. Default to None.
        :param callable, optional callback: Method called after evaluation. Default to None.
        """
        start_time = time.monotonic()
        budget = self.max_stage_duration if timeout is None else min(timeout, self.max_stage_duration)
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
        context = hash_evaluation_context(
            candidate.pipeline.fingerprint(), candidate.metrics, candidate.main_metric,
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
        if self.keep_training_history and candidate.training_audit is not None:
            return {
                "__computed_metrics__": deepcopy(candidate.computed_metrics),
                "__training_audit__": deepcopy(candidate.training_audit),
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
        self.candidates = self.first_step.run(candidate)

        return self.candidates

    ########################
    #### CONFIGURATIONS ####
    ########################

    def json_pipeline(self) -> dict:
        """Create a dictionary (JSON) from the loaded Pipeline

        :return: Loaded Pipeline in a JSON format
        """
        return self.first_step.json_pipeline()

    def all_configurations(self) -> list[dict]:
        """Return a dict with configurations of all steps. 

        :return: Configurations of all steps. 
        """
        return self.first_step.all_configurations()

    def configure_all(self, configs: dict) -> None:
        """Configure one to many steps with a dict configuration

        :param dict configs: key is a step_id and value is the configuration to set.
        """
        all_steps = self.__all_steps()

        for step_id, config in configs.items():
            current_step: Step = self.__find_step_by_id(all_steps, step_id)
            if current_step:
                for key, value in config:
                    current_step.configure(key, value)  # pylint: disable=no-member

    def __all_steps(self) -> list[Step]:
        """Recursive method. Return all the pipeline's steps in a list

        :return: All flatten pipelines's steps
        """
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
