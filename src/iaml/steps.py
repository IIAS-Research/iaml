"""Readable names for the existing IAML training adapters.

These are references to the historical classes, not additional registrations.
"""
from . import actionables as _actionables
from .step import Step as _Step

for _name, _component in vars(_actionables).items():
    if _name.startswith("Act") and isinstance(_component, type) and issubclass(_component, _Step):
        globals()[_name[3:]] = _component

# Historical names that describe a shorter algorithm rather than its sklearn
# estimator spelling remain available under both readable names.
RandomForestClassifier = _actionables.ActRandomForest
KNeighborsClassifier = _actionables.ActKNN
RandomForestRegressor = _actionables.ActRandomForestRegressor

# Scale each numeric row to unit L1/L2 norm. Normalizer remains a compatible alias.
UnitNormScaler = _actionables.ActUnitNormScaler

__all__ = [name for name in globals() if not name.startswith("_")]
