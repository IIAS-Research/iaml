from .step import Step, isStep, runner
from .metastep import MetaStep

@isStep('meta')
class MetaExplorerStep(MetaStep):
    # Explore all steps
    @runner
    def run(self, input, callback=None):
        output = []
        for step in self.steps:
            output = output + (step.run(input, callback=callback))
        
        return output
            