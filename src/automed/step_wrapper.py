from .step import Step, is_step, runner


@is_step('wrapper')
class StepWrapper(Step):
    def __init__(self, step: Step):
        self.step = step
        
    # Load any kind of StepWrapper
    @classmethod
    def from_pipeline(cls, pipeline:dict):
        if 'children' not in pipeline or len(pipeline['children']) != 1:
            raise TypeError(f'invalid pipeline: StepWrapper must have exactly one child')
        
        child = Step.from_pipeline(pipeline['children'][0])
        step = super().from_pipeline(pipeline, child)
        
        return step

    
    def configure_parents(self, *parents):
        self.step.configure_parents(*parents)
        
        super().configure_parents(*parents)
        

    def wrap(self, step):
        if Step in step.__class__.__mro__:
            self.step = step
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")
        
    # Is suitable if the wrapped step is
    def suitable(self, input):
        return self.step.suitable(input)
    
    def all_configurations(self):
        to_return = Step.all_configurations(self)
        to_return = to_return + self.step.all_configurations()
        
        return to_return
    
    
    def json_pipeline(self):
        return {
            'step': self.__class__.__name__,
            'name': self.name,
            'description': self.description,
            'configuration': self.configurations[0],
            'children': [self.step.json_pipeline()]
        }
    
    
    # Recursive function to get all steps in a pipeline
    def all_step(self):
        return [self.step]
        

    @runner
    def run(self, input_data, callback=None):
        # This wrapper is useless. Only run the step
        return self.step.run(input_data, callback=callback) 
    
    def count_steps(self):
        """
        Returns a rough estimation of the total count of steps for a given
        pipeline.
        """
        return 1 + self.step.count_steps()
    
    def priorize(self, input_data=None):
        return self.step.priorize(input_data)
            