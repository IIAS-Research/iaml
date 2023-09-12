from .step import Step, isStep, runner

@isStep('meta')
class MetaStep(Step):
    def __init__(self, tag=None):
        self.steps = []
        if tag:
            self.add_step_by_tag(tag) 
    
    
    def add_step(self, step):
        if Step in step.__class__.__mro__:
            self.steps.append(step)
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")
        
    def add_step_by_tag(self, tag):
        steps_to_add = set(filter(lambda key: tag in Step.available_steps[key], Step.available_steps.keys()))
        for step in steps_to_add:
            self.add_step(step())
    
    
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, input, callback=None):
        steps_to_run = self.steps.copy()
        
        current_input = input
        
        while steps_to_run:
            max_eval = steps_to_run[0].priorize(current_input)
            max_index = 0
            for index, step in enumerate(steps_to_run[1:]):
                current_eval = step.priorize(current_input)
                if current_eval > max_eval:
                    max_eval = current_eval
                    max_index = index+1
            
            current_input = steps_to_run[max_index].run(current_input, callback=callback)
            del steps_to_run[max_index]
        
        return current_input
            