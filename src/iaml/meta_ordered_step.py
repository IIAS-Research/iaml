"""
    Group several Step and run them in list order
"""
from typing import TYPE_CHECKING

from .decorators.all import is_step, runner
from .metastep import MetaStep

if TYPE_CHECKING:
    from .candidate import Candidate

#
# Inherit from MetaStep but will execute all steps without priorize() method.
#
@is_step('meta')
class MetaOrderedStep(MetaStep):
    """
    Group several Step and run them in list order
    """
    _usage: str = "Use when you need strict list-ordered execution of child steps instead of MetaStep prioritization. Applicable to pipelines where each step must run on the same Candidate in order. Avoid when steps are interchangeable and best-of exploration is needed (MetaExplorerStep)."
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, candidate:'Candidate') -> 'Candidate':
        current_candidate: Candidate = candidate
        for step in self.steps:
            current_candidate = step.run(current_candidate)

        return current_candidate
