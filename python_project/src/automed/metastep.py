from .logger import logger
from .step import Step, isStep, runner, Output, Input
from .step_wrapper import *

#
# MetaStep is a direct child of Step and will carry and execute several other Steps
# -> MetaStep will execute Step one by one, using the output of a step as input of the next one. 
# The order of Step is defined by the priorize() method 
#
# There is children classes of MetaStep to execute Steps in a different way
#
@isStep('meta')
class MetaStep(Step):
    name = "MetaStep"
    def __init__(self, tag=None, wrap=None, *args, **kw):
        self.steps = [] # Initialize steps to empty
        
        # If there is a tag -> add all Steps with this tag
        if tag:
            self.add_step_by_tag(tag, wrap=wrap) 
        
    # Load any MetaStep from json pipeline
    @classmethod
    def from_pipeline(cls, pipeline:dict):
        step = super().from_pipeline(pipeline)
        
        if 'children' in pipeline:
            for child in pipeline['children']:
                step.add_step(Step.from_pipeline(child))
                
        if 'tag' in pipeline:
            step.add_step_by_tag(pipeline['tag'])
            
        if not step.steps:
            raise TypeError(f'invalid pipeline: MetaStep must have at least one child')
        
        return step
    
    # Add one step to the MetaStep. 
    # step must be a Step inherited class
    def add_step(self, step):
        if Step in step.__class__.__mro__:
            self.steps.append(self.configure_child(step))
        else:
            raise Exception("step must be an occurence of step (or inherited classes)")
        
    # Add a list of Steps
    def add_steps(self, steps_list):
        for step in steps_list:
            self.add_step(step)
        
    # Add all Step with this tag to the MetaStep
    # Wrap -> It exist Will Wrap Steps with it. Check WrapperStep to know more 
    def add_step_by_tag(self, tag, wrap=None):
        steps_to_add = set(filter(lambda key: tag in Step.available_steps[key], Step.available_steps.keys()))
        
        if wrap != None:
            steps_to_add = list(map(lambda step: wrap(step()), steps_to_add))
        else:
            steps_to_add = list(map(lambda step: step(), steps_to_add))
        
        for step in steps_to_add:
            self.add_step(step);
            
    
    # Get configurations of all the steps and children steps 
    def all_configurations(self):
        to_return = Step.all_configurations(self)
        
        for step in self.steps:
            to_return = to_return + step.all_configurations()
            
        return to_return
    
    
    def json_pipeline(self):
        
        json = Step.json_pipeline(self)
        if any(self.steps):
            child = self.steps[-1].json_pipeline()
            for step in self.steps[-2::-1]:
                prev_child = child
                child = step.json_pipeline()
                
                child = self.__add_children_pipeline(child, [prev_child])
            
            json['children'] = [child]
        
        return json
    
    def __add_children_pipeline(self, pipeline, children):
        if ('children' in pipeline.keys()) and any(pipeline['children']):
            for index, child in enumerate(pipeline['children']):
                pipeline['children'][index] = self.__add_children_pipeline(child, children)
        else:
            pipeline['children'] = children
        
        return pipeline
                
    
    
    # Recursive function to get all steps in a pipeline
    def all_step(self):
        return self.steps
    
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, input, callback=None):
        results = []
        steps_to_run = self.steps.copy() # TODO copy is usefull ?
        
        return self.__recursive_run(steps_to_run, [input], callback=callback)
        
    
    # Recursive_run to manage Step with several outputs 
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


    def conf_to_rich_str_list(self):
        conf = super().conf_to_rich_str_list()
        conf.append(f'steps={",".join(set([ step.__class__.__name__ for step in self.steps ]))}')

        return conf
