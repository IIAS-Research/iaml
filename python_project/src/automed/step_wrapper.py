from .step import Step, isStep, runner

@isStep('wrapper')
class StepWrapper(Step):
    def __init__(self, step):
        self.step = step
        
    def wrap(self, step):
        if Step in step.__class__.__mro__:
            self.step = step
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")

    @runner
    def run(self, input, callback=None):
        # This wrapper is useless. Only run the step
        return self.step.run(input, callback=callback) 
            