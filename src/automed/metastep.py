"""
MetaStep is a direct child of Step and will carry and execute several other Steps
-> MetaStep will execute Step one by one, using the output of a step as input of the next one. 
The order of Step is defined by the priorize() method 

There is children classes of MetaStep to execute Steps in a different way
"""
from .step import Step, is_step, runner, Output, Input
from .step_wrapper import StepWrapper

@is_step('meta')
class MetaStep(Step):
    """
    MetaStep is a direct child of Step and will carry and execute several other Steps
    -> MetaStep will execute Step one by one, using the output of a step as input of the next one. 
    The order of Step is defined by the priorize() method 

    There is children classes of MetaStep to execute Steps in a different way
    """
    name = "MetaStep"
    def __init__(self,  *args, tag=None, wrap=None, **kwargs):
        self.steps:list[Step] = [] # Initialize steps to empty
        
        # If there is a tag -> add all Steps with this tag
        if tag:
            self.add_step_by_tag(tag, wrap=wrap) 
        
    @classmethod
    def from_pipeline(cls, pipeline:dict, *args) -> Step:
        """
        Load any MetaStep from json pipeline

        Args:
            pipeline (dict): Pipeline in a JSON format

        Raises:
            TypeError: invalid pipeline: MetaStep must have at least one child

        Returns:
            Step: Step created from Json pipeline
        """
        step = super().from_pipeline(pipeline)
        
        if 'children' in pipeline:
            for child in pipeline['children']:
                step.add_step(Step.from_pipeline(child))
                
        if 'tag' in pipeline:
            step.add_step_by_tag(pipeline['tag'])
            
        if not step.steps:
            raise TypeError('invalid pipeline: MetaStep must have at least one child')
        
        return step
    
    def configure_parents(self, *parents) -> None:
        """
        Back propagate child to parents
        """
        for step in self.steps:
            step.configure_parents(*parents)
        
        super().configure_parents(*parents)
    
    def add_step(self, step:Step) -> None:
        """
        Add one step to the MetaStep. 
        step must be a Step inherited class

        Args:
            step (Step): Step to add

        Raises:
            ValueError: step must be an occurrence of step (or inherited classes)
        """
        if Step in step.__class__.__mro__:
            self.steps.append(self.configure_child(step))
        else:
            raise ValueError("step must be an occurrence of step (or inherited classes)")
        
    def add_steps(self, steps:list[Step]) -> None:
        """
        Add a list of Steps

        Args:
            steps_list (list[Step]): _description_
        """
        for step in steps:
            self.add_step(step)
        
    def add_step_by_tag(self, tag:str, wrap:StepWrapper=None) -> None:
        """
        Add all Step with this tag to the MetaStep
        Wrap -> If exist, will wrap Steps with it. Check WrapperStep to know more 

        Args:
            tag (str): tag to search Steps
            wrap (StepWrapper, optional): Wrap Step in it. Defaults to None.
        """
        steps_to_add = Step.find_steps_by_tag(tag)
        
        if wrap is not None:
            steps_to_add = list(map(lambda step: wrap(step()), steps_to_add))
        else:
            steps_to_add = list(map(lambda step: step(), steps_to_add))
        
        for step in steps_to_add:
            self.add_step(step)
            
    def all_configurations(self) -> list[dict]:
        """
        Get configurations of all the steps and children steps 

        Returns:
            list[dict]: All configurations
        """
        to_return = Step.all_configurations(self)
        
        for step in self.steps:
            to_return = to_return + step.all_configurations()
            
        return to_return
    
    
    def json_pipeline(self) -> dict:
        """
        Create JSON pipeline

        Returns:
            dict: Pipeline in JSON format
        """
        return {
            **Step.json_pipeline(self),
            'children': [ step.json_pipeline() for step in self.steps ]
        }

    
    def all_step(self) -> Step:
        """
        Recursive function to get all steps in a pipeline

        Returns:
            list[Step]: Children steps
        """
        return self.steps
    
    @runner
    def run(self, input_data:Input, callback:callable=None) -> Output:
        """
        Run steps self ordered by "priorize" function

        Args:
            input_data (Input): Imput data
            callback (callable, optional): Call after each step run. Defaults to None.

        Returns:
            Output: Results
        """
        steps_to_run = self.steps.copy()
        
        return self.__recursive_run(steps_to_run, [input_data], callback=callback)
        
    
    def __recursive_run(self, remain_steps:list[Step], inputs:Input, callback=None) -> list[Output]:
        """
        Recursive_run to manage Step with several outputs 

        Args:
            remain_steps (list[Step]): Remaining Steps
            inputs (Input): Output of previous Step
            callback (_type_, optional): Call after each step run. Defaults to None.

        Returns:
            list[Output]: Results
        """
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
        return inputs

    def conf_to_rich_str_list(self) -> str:
        """
        Create a rich format string to describe step

        Returns:
            str: Step in rich format
        """
        conf = super().conf_to_rich_str_list()
        conf.append(f'steps={",".join({ step.__class__.__name__ for step in self.steps })}')

        return conf
    
    def count_steps(self) -> int:
        """
        Returns a rough estimation of the total count of steps for a given
        pipeline.
        """
        return 1 + sum(map(lambda child: child.count_steps(), self.steps))
