"""
    AutoMed is an autoML tools focusing on Medical Dataset with explainable models  
"""

import pandas as pd

from .step import Step
from .metastep import MetaStep
from .output import Input, Output
from .dataset import Dataset
from .metric import Metric
# from .destroyer import Destroyer
from .worker_manager import WorkerManager
from .wrapper.dataset import WrapKFold
from .meta_ordered_step import MetaOrderedStep
from .meta_explorer_step import MetaExplorerStep

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
class AutoMed:
    """ Main class of the module.
    AutoMed will load, configure and fit machine learning pipelines

    Attributes:
        output (list): Outputs of the pipeline after run
        fit_input (Input): last Input sent to the first step 
    """
    def __init__(self, max_workers:int=None, quiet:bool=False):
        self.outputs:list[Output] = None
        self.fit_input:Input = None
        self.first_step:Step = None # Will be the first Step of the pipeline (probably a MetaStep)
        
        Logger().set_quiet(quiet)
        self.default_pipeline() # Load default pipeline
        WorkerManager(max_workers=max_workers)

    def load_pipeline(self, pipeline:dict) -> None:
        """Load any kind of pipeline

        Args:
            pipeline (dict): JSON description of the pipeline
        """
        self.first_step = Step.from_pipeline(pipeline)

    # DEBUG -> Testing purpose
    def autosklearn_pipeline(self, time:int=30) -> None:
        """DEBUG -> Testing purpose
        Load a basic pipeline with AutoSkLearn as model 

        Args:
            time (int, optional): Max running time of AutoSKLearn model in seconds. Defaults to 30.
        """
        self.first_step = MetaOrderedStep()
        sklearn = ActAutoSKLearn() # pylint: disable=undefined-variable
        sklearn.configure('running_time', time)

        self.first_step.add_step(WrapKFold(sklearn))

    # DEBUG -> Testing purpose
    def tplot_pipeline(self) -> None:
        """DEBUG -> Testing purpose
        Load a basic pipeline with AutoSkLearn as model 
        """
        self.first_step = MetaOrderedStep()
        self.first_step.add_step(MetaStep(tag='cleaning'))
        self.first_step.add_step(MetaStep(tag='normalize'))
        self.first_step.add_step(ActTPLOT()) # pylint: disable=undefined-variable

    # DEBUG -> Testing purpose.
    def default_pipeline(self, use_destroyer=False, fast=False) -> None:
        """Load the default pipeline.
        Default pipeline is the recommended way to create classifier and regressor

        Args:
            use_destroyer (bool, optional): Enable/Disabled destroyer. 
                                            Destroyer doesn't work yet. Defaults to False.
            fast (bool, optional): If true, will only load fast machine learning model.
                                    Fast mode is use to create fast pipeline and iterate
                                    quickly when debugging code. Defaults to False.
        """
        self.first_step = MetaOrderedStep() # First step -> Contain all stages of the pipeline

        self.first_step.add_step(MetaStep(tag='cleaning'))
        self.first_step.add_step(MetaStep(tag='features_selection'))
        self.first_step.add_step(MetaStep(tag='normalize'))

        learning_tag = 'fast_learning' if fast else 'learning'

        def wrap(step: Step) -> 'WrapGeneticGridSearch':
            return WrapGeneticGridSearch(WrapKFold(step))
        
        if use_destroyer:
            self.first_step.add_step(MetaExplorerStep(tag=learning_tag,
                                                    wrap=wrap))
        else:
            self.first_step.add_step(MetaExplorerStep(tag=learning_tag, wrap=wrap))


    ##################
    ### PROPERTIES ###
    ##################

    # Dataset from input data
    @property
    def dataset(self) -> Dataset:
        """Shortcut to get input Dataset

        Returns:
            Dataset: Input dataset defined by .fit()
        """
        return self.fit_input.dataset

    ###########
    ### RUN ###
    ###########

    def fit(self, X:pd.DataFrame, y:list|pd.DataFrame, *args, **kwargs) -> list[Output]:
        """Run Pipeline to fit steps and models on X & y data. 
        
        Args:
            X (pd.DataFrame): Training features 
            y (pd.DataFrame): Training labels

        Returns:
            list[Output]: List of all the generated outputs. Sorted by performances.
        """
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        
        dataset:Dataset = Dataset(X, y)
        self.fit_input:Input = Input(dataset)

        # Select metrics used to evaluate performances
        for metric in self.__metrics_selection(X, y, dataset.type_of_target):
            self.fit_input.add_metric(metric)

        return self.__run(self.fit_input, *args, **kwargs)

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

    # Execute all the pipeline steps
        # Callback -> Will be call after each step 
    def __run(self, input_data:Input, callback:callable=None) -> list[Output]:
        """Run pipeline

        Args:
            input_data (Input): Data used to fit models and steps
            callback (callable, optional): Will be call after each Step run (-> many times).
                                            Defaults to None.

        Returns:
            list[Output]: List of all the generated outputs. Sorted by performances.
        """
        with Logger().progress as progress:
            step_count:int = self.first_step.count_steps()
            task = progress.add_task('Running...', total=step_count)
            
            # Override callback to handle progress bar
            def progress_callback(step: Step) -> None:
                progress.update(task, advance=1)
                if callback is not None:
                    callback(step)
                    
            # RUN!
            self.outputs = self.first_step.run(input_data, callback=progress_callback)

            progress.update(task, completed=step_count)

        # Order ouputs according the first metric
        self.outputs.sort(reverse=True)
        return self.outputs

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
