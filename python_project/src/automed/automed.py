from .step import *
from .metastep import MetaStep
from .actionable import Actionable
from .output import Output
from .dataset import Dataset
from .metric import Metric

from .meta_ordered_step import MetaOrderedStep

# Default Actionables
from .actionables.cleaning.act_mean_column import ActMeanColumn
from .actionables.cleaning.act_drop_numerical_column import ActDropNumericalColumn
from .actionables.cleaning.act_drop_textual_column import ActDropTextualColumn
from .actionables.cleaning.act_onehot import ActOnehot


from .actionables.learning.act_autosklearn import ActAutoSkLearn
from .actionables.learning.act_randomforest import ActRandomForest


from .actionables.random_split import RandomSplit

class AutoMed:
    output = None
    input:Dataset = None
    
    def __init__(self, dataset:Dataset=None):
        self.input = Output(dataset, None, None)
        self.first_step = None
    
    def load_pipe(self, pipe):
        pass
    
    def debug_load(self, only=None):
        if only:
            self.first_step = MetaStep(tag=only)
        else: 
            self.first_step = MetaOrderedStep()
            self.first_step.add_step(RandomSplit())
            self.first_step.add_step(MetaStep(tag='cleaning'))
            self.first_step.add_step(MetaStep(tag='learning'))
            
    @property
    def dataset(self):
        return self.input.dataset
    @property
    def metric(self):
        return self.input.metric
    @property
    def model(self):
        return self.input.model
    
    def run(self, callback=None):
        copied_input = self.input.to_output() # Avoid input to be edited by steps
        self.output = self.first_step.run(copied_input, callback=callback)
        return self.output
    
    