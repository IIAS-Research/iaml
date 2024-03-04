from .logger import Logger
from .step import Step, is_step, runner, Output, Input
from .step_wrapper import *

#
# MetaStep is a direct child of Step and will carry and execute several other Steps
# -> MetaStep will execute Step one by one, using the output of a step as input of the next one. 
# The order of Step is defined by the priorize() method 
#
# There is children classes of MetaStep to execute Steps in a different way
#
@is_step('meta')
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
    
    def configure_parents(self, *parents):
        for step in self.steps:
            step.configure_parents(*parents)
        
        super().configure_parents(*parents)
    
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
        steps_to_add = Step.find_steps_by_tag(tag)
        
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
        return {
            **Step.json_pipeline(self),
            'children': [ step.json_pipeline() for step in self.steps ]
        }

    
    # Recursive function to get all steps in a pipeline
    def all_step(self):
        return self.steps
    
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, input_data:Input, callback:callable=None):
        results = []
        steps_to_run = self.steps.copy() # TODO copy is usefull ?
        
        return self.__recursive_run(steps_to_run, [input_data], callback=callback)
        
    
    # Recursive_run to manage Step with several outputs 
    def __recursive_run(self, remain_steps, inputs, callback=None):
        outputs = []
        if remain_steps:
            for current_input in inputs:
                max_eval = remain_steps[0].priorize(current_input)
                max_index = 0
                for index, step in enumerate(remain_steps[1:]):
                    current_eval = step.priorize(current_input)
                    if current_eval > max_eval:
                        max_eval = current_eval 
                        max_index = index+1
                        
                results = remain_steps[max_index].run(current_input, callback=callback)
                futures_steps = remain_steps.copy()
                futures_steps.pop(max_index)
                outputs = outputs + self.__recursive_run(futures_steps, results, callback=callback)
                
            return outputs
        else:
            return inputs


    def conf_to_rich_str_list(self):
        conf = super().conf_to_rich_str_list()
        conf.append(f'steps={",".join(set([ step.__class__.__name__ for step in self.steps ]))}')

        return conf
    
    def count_steps(self):
        """
        Returns a rough estimation of the total count of steps for a given
        pipeline.
        """
        return 1 + sum(map(lambda child: child.count_steps(), self.steps))
