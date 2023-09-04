from step import Step, isStep

@isStep('meta')
class MetaStep(Step):
    steps = []
    
    
    def __init__(self):
        pass
    
    
    def add_step(self, step):
        if Step in step.__class__.__mro__:
            self.steps.append(step)
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")
        
    def add_step_by_tag(self, tag):
        steps_to_add = set(filter(lambda key: tag in Step.available_steps[key], Step.available_steps.keys()))
        for step in steps_to_add:
            self.add_step(step())
    
    
    # Run steps self ordered by "evaluate" function
    def run(self, dataset):
        steps_to_run = self.steps.copy()
        
        current_dataset = dataset
        while steps_to_run:
            max_eval = steps_to_run[0].evaluate(current_dataset)
            max_index = 0
            for index, step in enumerate(steps_to_run[1:]):
                current_eval = step.evaluate(current_dataset)
                if current_eval > max_eval:
                    max_eval = current_eval
                    max_index = index
                    
            current_dataset = steps_to_run[max_index].run(current_dataset)
            del steps_to_run[max_index]
        
        return current_dataset
            