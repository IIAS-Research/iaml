"""Classic kind of Step that transform, resample or predict from Candidate
"""

# -> Must be a wildcard import to help IAML to know all available the steps 
from .step import * # pylint: disable=unused-wildcard-import,wildcard-import
from .decorators.all import is_step

@is_step('actionable')
class Actionable(Step):
    """
    Classic kind of Step that transform, resample or predict from Candidate
    """
    def __init__(self, *args, **kwargs):
        self.configuration:dict = {}
