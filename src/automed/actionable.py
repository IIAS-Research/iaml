"""Classic kind of Step that transform, resample or predict from Input
"""

# -> Must be a wildcard import to help AutoMed to know all available the steps 
from .step import * # pylint: disable=unused-wildcard-import,wildcard-import

@isStep('actionable')
class Actionable(Step):
    """
    Classic kind of Step that transform, resample or predict from Input
    """
    def __init__(self, *args, **kw):
        self.configurations = [{}]
