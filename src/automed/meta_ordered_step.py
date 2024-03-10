"""
    Group several Step and run them in list order
"""
from typing import TYPE_CHECKING

from .decorators.all import is_step, runner
from .metastep import MetaStep

if TYPE_CHECKING:
    from .output import Input, Output

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
    def run(self, input_data:'Input', callback:callable=None) -> 'Output':
        """
        Run all step in order. Input will be transform successively by Steps

        Args:
            input_data (Input): Input to transform
            callback (callable, optional): Method to call after each Step. Defaults to None.

        Returns:
            Output: transformed data
        """
        current_input:Input = input_data
        for step in self.steps:
            current_input = step.run(current_input, callback=callback)
        
        return current_input
            