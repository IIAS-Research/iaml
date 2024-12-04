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
    # Run steps self ordered by "priorize" function
    @runner
    def run(self, candidate:'Candidate') -> 'Candidate':
        current_candidate: Candidate = candidate
        for step in self.steps:
            current_candidate = step.run(current_candidate)

        return current_candidate
