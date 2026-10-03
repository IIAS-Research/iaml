"""Behavioral contracts for declarative recipe editing and reconstruction."""
import pickle
import unittest
from types import SimpleNamespace
from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from iaml import Step
from iaml.flow import Const, Float, Int
from iaml.steps import (DecisionTreeClassifier, RandomForestClassifier, StandardScaler,
                        RobustScaler, MinMaxScaler, SMOTE, SMOTETomek, SMOTEENN,
                        SimpleImputer, Normalizer, UnitNormScaler)
from iaml.metrics import RecallMetric
from iaml.statistics import TopKValueCountsStatistic


class FlowRecipeTests(unittest.TestCase):















    def test_domains_validate_types_and_initial_values(self):
        with self.assertRaises(ValueError):
            Int(2, 1, initial=1)
        with self.assertRaises(TypeError):
            Int(1, 3, initial=True)
        with self.assertRaises(ValueError):
            Float(0, 1, initial=float("nan"))














if __name__ == "__main__":
    unittest.main()
