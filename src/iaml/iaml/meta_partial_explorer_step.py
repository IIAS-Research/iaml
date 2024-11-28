"""
[METASTEP] Generate pipeline without any exploration but optimizer 
        will be able to mutate into other steps
"""
from .meta_explorer_step import MetaExplorerStep
from .void_step import VoidStep
from .step import Step
from .decorators.all import is_step

#
# Inherit from MetaStep but will execute all steps at the same time. 
# The goal here is to explore many answer to a question. For example -> Try all Learning models
#
@is_step('meta')
class MetaPartialExplorerStep(MetaExplorerStep):
    """
    [METASTEP] Generate pipeline without any exploration but optimizer will 
    be able to mutate into other steps
    """
    name = "MetaPartialExplorerStep"
    
    def __init__(self, *args, tag: set=None, **kwargs):  # pylint: disable=unused-argument
        if steps := Step.find_steps_by_tag(tag):
            self.steps = [VoidStep(step_to_mimic=list(steps)[0]())]
