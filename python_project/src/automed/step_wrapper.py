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
        
    
    def all_configurations(self):
        to_return = Step.all_configurations(self)
        to_return = to_return + self.step.all_configurations()
        
        return to_return
    
    
    # Recursive function to get all steps in a pipeline
    def all_step(self):
        return [self.step]
        

    @runner
    def run(self, input, callback=None):
        # This wrapper is useless. Only run the step
        return self.step.run(input, callback=callback) 
            