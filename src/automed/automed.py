"""
    AutoMed is an autoML tools focusing on Medical Dataset with explainable models  
"""

import time
import multiprocessing
import pandas as pd
from .timed_pool_executor import TimedPoolExecutor
from .step import Step
from .cache import Cache
from .metastep import MetaStep
from .candidate import Candidate
from .dataset import Dataset
from .metric import Metric
from .worker_manager import WorkerManager
from .splitter import kfold_splitter
from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep
from .optimizers import Optimizer, GeneticOptimizer
from .meta_predictor import MetaPredictor

# Default Actionables -> Must be a wildcard import to help AutoMed to know all available the steps 
from .actionables import * # pylint: disable=unused-wildcard-import,wildcard-import

# Default Wrappers -> Must be a wildcard import to help AutoMed to know all available the steps 
from .wrapper import * # pylint: disable=unused-wildcard-import,wildcard-import

# cuDF pandas acceleration
try:
    import cudf.pandas 
    cudf.pandas.install()
    Logger().log('cuDF is installed: using cuDF pandas accelerator mode.')
except ImportError as e:
    Logger().log('cuDF not found: falling back to standalone pandas.')

# Main class of the package
# Useful to create & run pipeline
class AutoMed:  # pylint: disable=too-many-instance-attributes
    """ Main class of the module.
    AutoMed will load, configure and fit machine learning pipelines

    Attributes:
        candidate (list): Candidates of the pipeline after run
        fit_candidate (Candidate): last Candidate sent to the first step 
    """
    def __init__(self, # pylint: disable=too-many-arguments
                max_workers:int=None,
                quiet:bool=False,
                max_stage_duration:int=None,
                metalearner:bool=None,
                splitter=None,
                max_duration:int=-1,
                preprocessor:bool=False):
        
        Logger().set_quiet(quiet)
        
        self.preprocessor = preprocessor
        
        # Enable / Disable Meta Learner
        self.metalearner = metalearner
        if metalearner is None:
            if max_duration < 500 and max_duration != -1:
                Logger().log("Max duration under 500 seconds : \
                    Meta learner are disabled (you can enable it, \
                    with the parameter 'metalearner')")
                self.metalearner = False
            
        # Set max duration of each stage
        if max_stage_duration is None:
            self.max_stage_duration = max(max_duration / 5, 300)
            Logger().log(f"Max duration of each stage was set to {self.max_stage_duration} seconds")
        else:
            self.max_stage_duration = max_stage_duration

        # Set splitter
        self.splitter = splitter if splitter is not None else kfold_splitter
        
        
        self.max_duration = max_duration
        self.candidates:list[Candidate] = None
        self.fit_candidate:Candidate = None
        self.first_step:Step = None # Will be the first Step of the pipeline (probably a MetaStep)
        
        self.executor = None
        
        self.default_pipeline() # Load default pipeline
        self.max_workers = max_workers if (max_workers is not None and max_workers > 0 ) \
            else multiprocessing.cpu_count()
        
        self.chosen_candidate:Candidate = None
        WorkerManager(max_workers=self.max_workers)

    def __del__(self):
        del self.executor

    def load_pipeline(self, pipeline:dict) -> None:
        """Load any kind of pipeline

        Args:
            pipeline (dict): JSON description of the pipeline
        """
        self.first_step = Step.from_pipeline(pipeline)

    def default_pipeline(self, fast=False) -> None:
        """Load the default pipeline.
        Default pipeline is the recommended way to create classifier and regressor

        Args:
            fast (bool, optional): If true, will only load fast machine learning model.
                                    Fast mode is use to create fast pipeline and iterate
                                    quickly when debugging code. Defaults to False.
        """
        self.first_step = MetaOrderedStep() # First step -> Contain all stages of the pipeline

        self.first_step.add_step(MetaStep(tag='features_precleaning'))
        self.first_step.add_step(MetaStep(tag='cleaning'))
        self.first_step.add_step(MetaStep(tag='features_selection'))
        self.first_step.add_step(MetaStep(tag='normalize'))
        
        if self.preprocessor:
            self.first_step.add_step(
                MetaExplorerStep(tag='features_preprocessing', also_explore_without=True)
            )

        learning_tag = 'fast_predictor' if fast else 'predictor'
        
        self.first_step.add_step(MetaExplorerStep(tag=learning_tag))


    ##################
    ### PROPERTIES ###
    ##################

    # Dataset from candidate data
    @property
    def dataset(self) -> Dataset:
        """Shortcut to get candidate Dataset

        Returns:
            Dataset: Candidate dataset defined by .fit()
        """
        return self.fit_candidate.dataset

    ###########
    ### RUN ###
    ###########

    def fit(self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            *args,
            groups:pd.DataFrame = None,
            patience:int=-1,
            **kwargs) -> list[Candidate]:
        """Run Pipeline to fit steps and models on X & y data. 
        
        Args:
            X (pd.DataFrame): Training features 
            y (pd.DataFrame): Training labels

        Returns:
            list[Candidate]: List of all the generated candidates. Sorted by performances.
        """
        start_time = time.monotonic()
        
        self.executor = TimedPoolExecutor(max_workers=self.max_workers)
        
        try:
            if isinstance(y, pd.DataFrame):
                y = y.values.ravel()
                
            ### INITIAL GENERATE CANDIDATE 
            dataset:Dataset = Dataset(deepcopy(X), deepcopy(y), groups=groups)
            self.fit_candidate:Candidate = Candidate(dataset)

            # Select metrics used to evaluate performances
            for metric in self.__metrics_selection(dataset.X, dataset.y, dataset.type_of_target):
                self.fit_candidate.add_metric(metric)

            # Generate candidates
            candidates = self.__run(self.fit_candidate, *args, **kwargs)
         
            # Remove candidate without predictor 
            candidates = [candidate for candidate in candidates \
                if candidate.pipeline.predictor is not None]
            Logger().log(f"{len(candidates)} generated pipelines")
            
            ### INITIAL EVALUATION
            
            # Evaluate candidates
            candidates = self.__run_evaluations(candidates,
                            dataset,
                            timeout=self.max_duration - (time.monotonic() - start_time))
            
            ### FINETUNING
            candidates = self.__optimize(dataset,
                                        candidates,
                                        optimizer=GeneticOptimizer(),
                                        max_duration=self.max_duration - \
                                            (time.monotonic() - start_time),
                                        patience=patience)
            ### FINAL FIT
            
            self.executor.shutdown()
            
            # Fit candidate with the whole dataset
            Cache.reset()
            self.chosen_candidate = deepcopy(candidates[0])
            self.chosen_candidate.pipeline.fit(X, y)
            
            return candidates
        except Exception as e:
            print("Error during fit")
            self.executor.shutdown()
            raise e
    
    @property
    def chosen_model(self):
        """
        Return the best model trained with fit

        Returns:
            AutoPipeline: Best predictor pipeline
        """
        if not self.chosen_candidate:
            return None
        
        return self.chosen_candidate.pipeline
    
    def __run_evaluations(self,
                        candidates:Candidate,
                        dataset:Dataset,
                        timeout:int=None,
                        stage_number:int=None) -> None:
        
        new_candidates:list[Candidate] = []

        with Logger().progress as progress:
            task = progress.add_task(
                f'Stage {stage_number}' if stage_number is not None else "Initial evaluation",
                total=len(candidates))
            
            def update_progressbar(*args): # pylint: disable=unused-argument
                progress.update(task, advance=1)
            
            self.executor.set_callback(update_progressbar)
            for candidate in candidates:
                from_cache = Cache().from_cache( \
                    'automed_'+candidate.pipeline.fingerprint(), dataset.X)
                
                if from_cache:
                    candidate.computed_metrics = from_cache
                    new_candidates.append(candidate)
                    update_progressbar() # Update progressbar even if data come from cache
                else:
                    self.executor.submit(
                            process_executor,
                            candidate,
                            dataset,
                            splitter=self.splitter
                        )
            
            # Wait for all tasks to complete with a timeout
            new_candidates += self.executor.join(min(timeout, self.max_stage_duration))
            
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
            fingerprint = candidate.pipeline.fingerprint()
            if not Cache().from_cache('automed_'+fingerprint, dataset.X):
                Cache().add_to_cache('automed_'+fingerprint, dataset.X, candidate.computed_metrics)
                
        return new_candidates
        
            
    
    def __optimize(self,
                    dataset:Dataset,
                    candidates:list[Candidate],
                    optimizer:Optimizer=Optimizer(),
                    patience:int=5,
                    max_duration:int=-1) -> list[Candidate]:
        if not candidates:
            return []
        
        # init
        candidates.sort(reverse=True)
        best_result:float = candidates[0].get_main_metric_value()
        iterations_without_improvement:int = 0
        iterations_count:int = 0
        duration:int = 0
        starting_time:int = time.monotonic() # seconds
        
        # If there is not, define an arbitrary stop condition
        if max_duration == -1 and patience == -1:
            Logger().log('You have not defined any stop condition. \
                Patient has arbitrary set to 20')
            patience = 20
        
        while   not(optimizer.finished) \
                and (patience == -1 or iterations_without_improvement < patience) \
                and (max_duration == -1 or max_duration > duration):
            # Generate new candidates
            candidates = optimizer.run(candidates)
            
            if self.metalearner:
                # Generate metapredictor
                if len(candidates) > 1:
                    for metapredictor in self.__meta_predictor_iter(dataset.type_of_target):
                        meta_candidate:MetaPredictor = metapredictor(
                            [candidate for candidate in candidates if not candidate.is_meta][0:5]
                            ).to_candidate()
                        candidates.append(meta_candidate)
                
            Logger().log(f'Finetuning... \
                stage={iterations_count} \
                candidates={len(candidates)} \
                patience={iterations_without_improvement}/{patience}, \
                duration={round(duration, 2)}/{max_duration}, \
                best_result={best_result}')
            
            # Evaluate new candidates
            candidates = self.__run_evaluations(candidates,
                        dataset,
                        timeout=max_duration - (time.monotonic() - starting_time),
                        stage_number=iterations_count)
            
            # Remove not computed (error or timeout)
            candidates = [candidate for candidate in candidates if candidate.computed_metrics]
            
            # Logger().log([(round(candidate.get_main_metric_value(), 5), \
            #     candidate.pipeline.predictor[0], \
            #     candidate.pipeline.predictor[1].resume_configuration()) \
            #         for candidate in candidates], force=True)
            
            # Improvement ?
            new_best:float = candidates[0].get_main_metric_value()
            if new_best > best_result:
                best_result = new_best
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                
            # Duration in seconds
            duration = time.monotonic() - starting_time
            
            # Increase Iteration count
            iterations_count += 1
            
            
        return candidates[0:5]
        
        

    def __metrics_selection(self, X:pd.DataFrame, y:pd.DataFrame, type_of_target:str):
        """Select metrics used to evaluate performances

        Args:
            X (pd.DataFrame): Training features
            y (pd.DataFrame): Training labels
            type_of_target (str): type of label. Example : continuous, binary

        Returns:
            list[Metric]: List of selected metrics
        """
        metrics = []

        for metric_sub_class in Metric.__subclasses__():
            # Instantiate a subclass
            metric = metric_sub_class()
            # Verify if a subclass is suitable or not
            if metric.suitable(X, y, type_of_target):
                metrics.append(metric)
        return metrics
    
    def __meta_predictor_iter(self, type_of_target:str):
        for subclass in MetaPredictor.__subclasses__():
            # Verify if a subclass is suitable or not
            if subclass.suitable(type_of_target):
                yield subclass

    # Execute all the pipeline steps
        # Callback -> Will be call after each step 
    def __run(self, candidate:Candidate, callback:callable=None) -> list[Candidate]:
        """Run pipeline

        Args:
            candidate (Candidate): Data used to fit models and steps
            callback (callable, optional): Will be call after each Step run (-> many times).
                                            Defaults to None.

        Returns:
            list[Candidate]: List of all the generated candidates. Sorted by performances.
        """
        with Logger().progress as progress:
            step_count:int = self.first_step.count_steps()
            task = progress.add_task('Generate candidates', total=step_count)
            # Override callback to handle progress bar
            def progress_callback(step: Step) -> None:
                progress.update(task, advance=1)
                if callback is not None:
                    callback(step)
                    
            # RUN!
            self.candidates = self.first_step.run(candidate, callback=progress_callback)
            progress.update(task, completed=step_count)
            
        # Order candidate according the main metric
        self.candidates.sort(reverse=True)
        return self.candidates

    ########################
    #### CONFIGURATIONS ####
    ########################

    def json_pipeline(self) -> dict:
        """Create a dictionary (JSON) from the loaded Pipeline

        Returns:
            dict: Loaded Pipeline in a JSON format
        """
        return self.first_step.json_pipeline()

    def all_configurations(self) -> list[dict]:
        """Return a dict with configurations of all steps. 

        Returns:
            list[dict]: Configurations of all steps. 
        """
        return self.first_step.all_configurations()

    # Configure one to many steps with a dict configuration
    def configure_all(self, configs:dict) -> None:
        """Configure one to many steps with a dict configuration

        Args:
            configs (dict): key is a step_id and value is the configuration to set.
        """
        all_steps = self.__all_steps()

        for step_id, config in configs.items():
            current_step = self.__find_step_by_id(all_steps, step_id)
            if current_step:
                for key, value in config:
                    current_step.configure(key, value)

    def __all_steps(self) -> list[Step]:
        """ Recursive method. Return all the pipeline's steps in a list

        Returns:
            list[Step]: All flatten pipelines's steps
        """
        return self.first_step + self.first_step.all_steps()

    def __find_step_by_id(self, step_list:list[Step], step_id:int) -> Step:
        """_summary_

        Args:
            step_list (list[Step]): The step will be searched in this list
            step_id (int): Identifier of the step

        Returns:
            Step: Found Step or None
        """
        for step in step_list:
            if id(step) == step_id:
                return step
        return None


def process_executor(candidate:Candidate, *args, **kwargs) -> 'Candidate':
    """
    Wrap candidate training to run it in subprocess

    Args:
        candidate (Candidate): Not trained candidate

    Returns:
        Candidate: Trained candidate
    """
    # Deepcopy -> Without it, process end is never detected. Strange...
    candidate = deepcopy(candidate)
    candidate.training_evaluate(*args, **kwargs)
    return candidate
