from .step import Step, isStep, runner
from .metastep import MetaStep


#
# Inherit from MetaStep but will execute all steps without priorize() method. 
#
@isStep('meta')
class MetaOrderedStep(MetaStep):
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, input, callback=None):
        current_input = input
        for step in self.steps:
            current_input = step.run(current_input, callback=callback)
        
        return current_input
            