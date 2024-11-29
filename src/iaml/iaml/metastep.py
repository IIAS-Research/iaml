"""MetaStep is a direct child of Step and will carry and execute several other Steps
-> MetaStep will execute Step one by one, using the candidate of a step as candidate of 
the next one.

The order of Step is defined by the priorize() method .

There is children classes of MetaStep to execute Steps in a different way
"""
from typing import Any
from .step import Step
from .candidate import Candidate
from .decorators.all import find_steps_by_tag, is_step, runner
from .step_wrapper import StepWrapper


@is_step('meta')
class MetaStep(Step):
    """MetaStep is a direct child of Step and will carry and execute several other Steps
    -> MetaStep will execute Step one by one, using the candidate of a step as candidate of 
    the next one.

    The order of Step is defined by the priorize() method 

    There is children classes of MetaStep to execute Steps in a different way
    
    :param tuple, optional \\*args: Additional parameters.
    :param str, optional str: The tag of the meta step. Default to None.
    :param StepWrapper, optional wrap: If exist, will wrap Steps with it. Default to None.
    :param str, optional name: Name of the MetaStep. Default to None.
    :param str, optional description: Description of the MetaStep. Default to None.
    :param dict, optional \\**kwargs: Additional parameters.
    """
    name: str = "Steps group"
    description: str = 'Execute steps one by one'
    description_long: str = None

    def __init__(
        self,
        *args,
        tag: str = None,
        wrap: StepWrapper = None,
        name: str = None,
        description: str = None,
        **kwargs) -> None:
        self.steps:list[Step] = [] # Initialize steps to empty
        """List of steps"""

        # If there is a tag -> add all Steps with this tag
        if tag:
            self.add_step_by_tag(tag, wrap=wrap)
            self.name = f"Steps from : {tag}"

        if name:
            self.name = name
        if name:
            self.description = description

    @classmethod
    def from_pipeline(cls, pipeline: dict[str, Any], *args, **kwargs) -> Step:
        metastep = super().from_pipeline(pipeline)

        if 'children' in pipeline:
            for child in pipeline['children']:
                metastep.add_step(Step.from_pipeline(child))

        if 'tag' in pipeline:
            metastep.add_step_by_tag(pipeline['tag'])

        if not metastep.steps:
            raise TypeError('invalid pipeline: MetaStep must have at least one child')

        return metastep

    def configure_parents(self, *parents: list[Step]) -> None:
        for step in self.steps:
            step.configure_parents(*parents)

        super().configure_parents(*parents)

    def add_step(self, step: Step) -> None:
        """Add one step to the MetaStep. 
        step must be a Step inherited class

        :param Step step: Step to add
        :raise ValueError: step must be an occurrence of step (or inherited classes)
        """
        if Step in step.__class__.__mro__:
            step = self.configure_child(step)

            # Define enable only if False because True can generate strange behavior.
            # True is default value anyway
            if not self._enable:
                step.enable = False

            self.steps.append(step)
        else:
            raise ValueError("step must be an occurrence of step (or inherited classes)")

    def add_steps(self, steps: list[Step]) -> None:
        """Add a list of Steps

        :param list[Step] steps: List of steps to add to this MetaStep
        """
        for step in steps:
            self.add_step(step)

    def add_step_by_tag(self, tag: str, wrap: StepWrapper = None) -> None:
        """Add all Step with this tag to the MetaStep
        Wrap -> If exist, will wrap Steps with it. Check WrapperStep to know more 

        :param str tag: tag to search Steps.
        :param StepWrapper, optional wrap: Wrap Step in it. Default to None.
        """
        steps_to_add = find_steps_by_tag(tag)

        if wrap is not None:
            steps_to_add = list(map(lambda step: wrap(step()), steps_to_add))
        else:
            steps_to_add = list(map(lambda step: step(), steps_to_add))

        for step in steps_to_add:
            self.add_step(step)

    def all_configurations(self) -> list[dict[str, Any]]:
        to_return = Step.all_configurations(self)

        for step in self.steps:
            to_return = to_return + step.all_configurations()

        return to_return

    def json_pipeline(self) -> dict[str, Any]:
        return {
            **Step.json_pipeline(self),
            'children': [ step.json_pipeline() for step in self.steps ]
        }

    def all_steps(self) -> list[Step]:
        children = []
        for step in self.steps:
            children += step.all_steps()

        return [self, *children]

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        steps_to_run = self.steps.copy()
        return self.__recursive_run(steps_to_run, [candidate])


    def __recursive_run(self,
            remain_steps: list[Step],
            candidates: list[Candidate]) -> list[Candidate]:
        """Recursive_run to manage Step with several candidates 

        :param list[Step] remain_steps: Steps remaining.
        :param list[Candidate] candidates: Candidate of previous Step.
        :return: Results.
        """
        output_candidates = []
        if remain_steps:
            for current_candidate in candidates:
                max_eval = remain_steps[0].priorize(current_candidate)
                max_index = 0
                for index, step in enumerate(remain_steps[1:]):
                    current_eval = step.priorize(current_candidate)
                    if current_eval > max_eval:
                        max_eval = current_eval
                        max_index = index+1

                results = remain_steps[max_index].run(current_candidate)
                futures_steps = remain_steps.copy()
                futures_steps.pop(max_index)
                output_candidates = output_candidates + \
                    self.__recursive_run(futures_steps, results)

            return output_candidates
        return candidates

    def conf_to_rich_str_list(self) -> str:
        conf = super().conf_to_rich_str_list()
        conf.append(f'steps={",".join({ step.__class__.__name__ for step in self.steps })}')

        return conf

    def count_steps(self) -> int:
        return 1 + sum(map(lambda child: child.count_steps(), self.steps))


    @property
    def enable(self) -> bool:
        return any(step.enable for step in self.steps) if hasattr(self, 'steps') else False

    @enable.setter
    def enable(self, value: bool) -> bool:
        if hasattr(self, 'steps'):
            for step in self.steps:
                step.enable = value

        self._enable = value # Only used has default value for new children

        return self.enable
