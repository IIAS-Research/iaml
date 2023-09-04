from step import *
from metastep import MetaStep
# from orderedmetastep import OrderedMetaStep
from actionable import Actionable

# Default Actionables
from actionables.cleaning.act_mean_column import ActMeanColumn
from actionables.cleaning.act_drop_column import ActDropColumn

class AutoMed:
    output = None 
    
    def __init__(self, input):
        self.input = input
        self.first_step = None
    
    def load_pipe(self, pipe):
        pass
    
    def debug_load(self):
        self.first_step = MetaStep()
        # self.first_step.add_step(ActMeanColumn())
        # self.first_step.add_step(ActDropColumn())
        self.first_step.add_step_by_tag('cleaning')
        
    def run(self):
        self.output = self.first_step.run(self.input)
        return self.output
    
    