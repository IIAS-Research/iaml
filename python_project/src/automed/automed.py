from step import *
from metastep import MetaStep
from actionable import Actionable
from output import Output
from dataset import Dataset
from metric import Metric

from meta_ordered_step import MetaOrderedStep


# Default Actionables
from actionables.cleaning.act_mean_column import ActMeanColumn
from actionables.cleaning.act_drop_numerical_column import ActDropNumericalColumn
from actionables.cleaning.act_drop_textual_column import ActDropTextualColumn
from actionables.cleaning.act_onehot import ActOnehot
from actionables.learning.act_autosklearn import ActAutoSkLearn
from actionables.random_split import RandomSplit

class AutoMed:
    output = None 
    
    def __init__(self, dataset:Dataset):
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
    
    def run(self):
        self.output = self.first_step.run(self.input)
        return self.output
    
    