from .step import Step, isStep, runner, Output, Input
from .step_wrapper import *

@isStep('meta')
class MetaStep(Step):
    def __init__(self, tag=None, wrap=None, *args, **kw):
        self.steps = []
        if tag:
            self.add_step_by_tag(tag, wrap=wrap) 
    
    
    def add_step(self, step):
        if Step in step.__class__.__mro__:
            self.steps.append(self.configure_child(step))
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")
        
    def add_steps(self, steps_list):
        for step in steps_list:
            self.add_step(step)
        
    def add_step_by_tag(self, tag, wrap=None):
        steps_to_add = set(filter(lambda key: tag in Step.available_steps[key], Step.available_steps.keys()))
        
        if wrap != None:
            steps_to_add = list(map(lambda step: wrap(step()), steps_to_add))
        else:
            steps_to_add = list(map(lambda step: step(), steps_to_add))
        
        for step in steps_to_add:
            self.add_step(step);
            
    
    def all_configurations(self):
        to_return = Step.all_configurations(self)
        
        for step in self.steps:
            to_return = to_return + step.all_configurations()
            
        return to_return
    
    
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, input, callback=None):
        results = []
        steps_to_run = self.steps.copy() # TODO copy is usefull ?
        
        return self.__recursive_run(steps_to_run, [input], callback=callback)
        
    
    def __recursive_run(self, remain_steps, inputs, callback=None):
        if remain_steps:
            for input in inputs:
                max_eval = remain_steps[0].priorize(input)
                max_index = 0
                for index, step in enumerate(remain_steps[1:]):
                    current_eval = step.priorize(input)
                    if current_eval > max_eval:
                        max_eval = current_eval 
                        max_index = index+1
                        
            results = remain_steps[max_index].run(input, callback=callback)
            futures_steps = remain_steps.copy()
            futures_steps.pop(max_index)
            return self.__recursive_run(futures_steps, results, callback=callback)
        else:
            return inputs