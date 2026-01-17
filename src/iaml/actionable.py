"""Classic kind of Step that transform, resample or predict from Candidate"""

# -> Must be a wildcard import to help IAML to know all available the steps
from .step import * # pylint: disable=unused-wildcard-import,wildcard-import
from .decorators.all import is_step


@is_step('actionable')
class Actionable(Step):
    """Classic kind of Step that transform, resample or predict from Candidate"""
    _usage = "Use when you need a concrete transform/resample/predict step over a Candidate. Applicable to pipeline steps that operate directly on Candidate data. Avoid when you need meta orchestration like MetaStep or wrappers like StepWrapper."
