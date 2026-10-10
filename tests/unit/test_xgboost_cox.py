"""Essential checks for Cox labels and Breslow calibration."""
import unittest

import numpy as np
import pandas as pd
from xgboost import DMatrix

from iaml.actionables.predictors._xgboost import xgboost_features
from iaml.actionables.predictors.survival._cox_breslow import CoxBreslowBaseline
from iaml.actionables.predictors.survival.act_survival_xgboost import (
    ActSurvivalXGBoost, _cox_labels,
)
from iaml.dataset import Dataset


class TestXGBoostCox(unittest.TestCase):
    def test_native_labels_preserve_censoring_zero_times_and_float32_order(self):
        events = np.array([True, False, True, False])
        cases = (
            ([2, 2, 7, 11], [2, -2, 7, -11]),
            ([0, 0, 4, 8], [1, -1, 4, -8]),
            ([0, 0, 0, 0], [1, -1, 1, -1]),
            ([0, 0, 1e-300, 1e300], [1, -1, 2, -3]),
        )
        for times, expected in cases:
            with self.subTest(times=times):
                labels = _cox_labels(events, np.array(times, dtype=float))
                native = DMatrix(np.zeros((4, 1)), label=labels).get_label()
                np.testing.assert_array_equal(native, expected)

    def test_breslow_uses_tied_risk_sets_and_handles_the_origin(self):
        margins = np.log([1, 2, 3, 4, 5])
        events = np.array([True, True, False, True, False])
        times = np.array([0, 0, 0, 1, 2], dtype=float)
        # Both events at zero include the tied censor in their denominator.
        expected = np.array([2 / 15, 2 / 15, 2 / 15 + 1 / 9, 2 / 15 + 1 / 9])
        for origin in (0, 1):
            grid = np.array([0, 0.5, 1, 2]) + origin
            hazard = 2 * expected
            if origin:
                grid, hazard = np.r_[0, grid], np.r_[0, hazard]
            for shift in (0, 1000):
                with self.subTest(origin=origin, shift=shift):
                    baseline = CoxBreslowBaseline().fit(margins + shift, events, times + origin)
                    prediction = np.array([np.log(2) + shift])
                    cumulative = baseline.get_cumulative_hazard_function(prediction)[0]
                    survival = baseline.get_survival_function(prediction)[0]
                    np.testing.assert_allclose(cumulative(grid), hazard, rtol=1e-11)
                    np.testing.assert_allclose(survival(grid), np.exp(-hazard), rtol=1e-11)

    def test_label_adaptation_keeps_original_targets_and_curve_times(self):
        X = pd.DataFrame({'feature': np.arange(4, dtype=float)})
        y = [(True, 0), (False, 0), (True, 1e-300), (False, 1e300)]
        dataset = Dataset(X, y)
        original = dataset.y.copy()
        step = ActSurvivalXGBoost()
        step.configure({'n_estimators': 3})
        step.fit(dataset)
        np.testing.assert_array_equal(dataset.y, original)
        np.testing.assert_array_equal(step.unique_times_, [0, 1e-300, 1e300])
        risks = np.exp(step.model.predict(xgboost_features(X), output_margin=True).astype(float))
        initial = 1 / risks.sum()
        baseline = np.array([initial, initial + 1 / risks[2:].sum(), initial + 1 / risks[2:].sum()])
        curve = step.predict_cumulative_hazard_function(X)[2]
        np.testing.assert_allclose(curve(step.unique_times_), risks[2] * baseline)

    def test_invalid_training_targets_are_rejected(self):
        X = pd.DataFrame({'feature': [1, 2]})
        cases = (
            [], [(True, -1), (False, 2)], [(True, np.nan), (False, 2)],
            [(True, np.inf), (False, 2)], [(False, 1), (False, 2)], [(True, 1)],
        )
        for y in cases:
            with self.subTest(y=y), self.assertRaises(ValueError):
                ActSurvivalXGBoost().fit(Dataset(X, y))


if __name__ == '__main__':
    unittest.main()
