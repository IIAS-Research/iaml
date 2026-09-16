"""Compatibility import for the former survival boosting module.

The implementation lives in ``act_gradient_boosting_survival_analysis``.
Re-export the same class so existing imports and pickles keep resolving without
registering a second predictor. The backend is scikit-survival, not XGBoost.
"""
from .act_gradient_boosting_survival_analysis import ActGradientBoostingSurvivalAnalysis

__all__ = ['ActGradientBoostingSurvivalAnalysis']
