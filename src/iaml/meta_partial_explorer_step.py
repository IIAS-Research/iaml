"""[METASTEP] Generate pipeline without any exploration but optimizer 
will be able to mutate into other steps
"""
from .meta_explorer_step import MetaExplorerStep
from .step import Step
from .void_step import VoidStep
from .decorators.all import find_steps_by_tag, is_step

@is_step('meta')
class MetaPartialExplorerStep(MetaExplorerStep):
    """[METASTEP] Generate pipeline without any exploration but optimizer will 
    be able to mutate into other steps

    :param Step, optional initial_step: Run this step initially instead of a no-op.
        When a tag is supplied, the step must carry that tag.
    """

    name: str = "MetaPartialExplorerStep"
    _usage: str = "Use when you want one initial choice that the optimizer can later replace. Starts with initial_step when supplied, otherwise a no-op matching the tag. Avoid when you need immediate exploration or an optimizer that cannot replace steps."

    def __init__(self, *args, tag: str = None, initial_step: Step = None,
                 **kwargs):  # pylint: disable=unused-argument
        self.steps = []
        # Partial exploration always emits a single branch, including for a no-op.
        self.also_explore_without = False
        if initial_step is not None:
            if not isinstance(initial_step, Step):
                raise TypeError("initial_step must be a Step")
            if tag is not None and tag not in (initial_step.tags or set()):
                raise ValueError("initial_step must match the exploration tag")
            self.add_step(initial_step)
        elif steps := find_steps_by_tag(tag):
            representative = min(steps, key=lambda step: (step.__module__, step.__name__))
            self.add_step(VoidStep(step_to_mimic=representative()))
